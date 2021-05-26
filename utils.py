import functools
from altair.utils.core import infer_dtype
import networkx as nx
import numpy as np
import os, sys
import pandas as pd
import pickle as pkl
import streamlit as st
import types
from collections import defaultdict

from collections import defaultdict

from annotated_text import annotated_text


### Params

BIG_FONT_SIZE = 24
SMALL_FONT_SIZE = 18


BY_CLUSTER_PARAMS = ({
    'groupby': '# clusters',
    'sortby':  '# ads' 
}, {
    'y': '# ads',
    'facet': '# clusters:N',
    'tooltip': ['days', '# ads'],
    'show_labels': False
})

BY_METADATA_PARAMS = ({
    'groupby': 'metadata',
    'sortby':  'count' 
}, {
    'y': 'count',
    'facet': 'metadata:N',
    'tooltip': ['days',  'count', 'metadata', 'type'],
})

BUTTON_STYLE = '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>'


### Generic utils
@st.cache#(show_spinner=False)
def read_csv(filename, keep_cols=[], rename_cols={}):
    ''' read csv into Pandas DataFrame
        :param filename:    location of csv file
        :param keep_cols:   if specified, only keep those csv columns
        :param rename_cols: if specified, dictionary of rename mappings for csv columns
        :return:            Pandas DataFrame '''
    df = pd.read_csv(filename)
    if keep_cols:
        df = df[keep_cols]

    if rename_cols:
        df = df.rename(columns=rename_cols)

    return df


@st.cache(hash_funcs={types.GeneratorType: id}, show_spinner=False)
def get_subdf(df, state, date_col='day_posted'):
    ''' get subset of DataFrame based on state.cluster, do location & date processing
        :param df:          DataFrame to take subset of
        :param state:       SessionState object, state.cluster shows subset to take
        :param date_col:    name of DataFrame column containing date 
        :return:            subset of DataFrame with nicely formatted locatino & date'''
    subdf = df[df['LSH label'].isin(state.cluster)].copy()

    if 'location' in subdf.columns: # means pre-processed script already ran
        return subdf

    subdf = gen_locations(subdf)

    subdf['location'] = [prettify_location(*tup) for tup in subdf[['city_id', 'country_id']].values]

    subdf[date_col] = pd.to_datetime(subdf[date_col], infer_datetime_format=True)
    subdf[date_col] = subdf[date_col].dt.normalize()

    return subdf


#@st.cache(show_spinner=False)
def pre_process_df(df, filename, date_col='day_posted', use_cache=True):
    clean_filename = './data/{}-cleaned.csv'.format(os.path.splitext(os.path.basename(filename))[0])
    if use_cache and os.path.exists(clean_filename):
        copy_df = pd.read_csv(clean_filename)
        copy_df['days'] = pd.to_datetime(copy_df.days, infer_datetime_format=True)
        return copy_df

    copy_df = df.copy()
    copy_df['location'] = [prettify_location(*tup) for tup in df[['city_id', 'country_id']].values]

    days = pd.to_datetime(copy_df[date_col], infer_datetime_format=True)
    copy_df['days'] = days.dt.tz_localize('UTC', ambiguous=True)

    copy_df = gen_locations(copy_df)

    copy_df.to_csv(clean_filename, index=False)

    return copy_df


@st.cache#(show_spinner=False)
def extract_field(series):
    ''' extract values from Pandas Series, where some entries represent multiple values
        :param series:  Pandas Series to get values from
        :return:        Numpy 1D array of all values (with repetitions)
        '''
    series = series.dropna()
    if not len(series):
        return series

    return np.concatenate(series.apply(lambda val: str(val).split(';')).values)


@st.cache
def pretty_basic_stats(stats):
    ''' prettify output of basic_stats '''
    text = []
    for name, (count, unique) in stats.items():
        name = name.split()[1]
        if name == 'clusters':
            name = 'micro-clusters'

        if unique == '--':
            text.append('**{}** {}'.format(count, name))
            continue

        text.append('**{}** total {} (**{}** unique)'.format(count, name, unique))

    return ', '.join(text[:-1]) + ', and ' + text[-1] + '.'


@st.cache
def pretty_s(s):
    ''' prettify a string for display
        :param s:  string to prettify
        :return     string with spaces and plural '''
    return '# {}s'.format(s.replace('_', ' '))


@st.cache
def filename_stub(filename):
    ''' strip path and extension from filename
        :param filename:    string of filename
        :return:            stub of filename '''
    return os.path.basename(filename).split('.')[0]


@st.cache#(show_spinner=False)
def basic_stats(df, cols, cluster_label='LSH label'):
    ''' get basic meta-cluster level stats, not based on time
        :param df:      Pandas DataFrame representing ads from one meta-cluster
        :param cols:    columns of DataFrame containing relevant metadata
        :return:        DataFrame with metadata counts '''

    metadata = {pretty_s(col): [len(extract_field(df[col])), len(set(extract_field(df[col])))] for col in cols}
    metadata['# ads'] = [len(df), '--']
    metadata['# clusters'] = [len(df[cluster_label].unique()), '--']

    return metadata


@st.cache
def top_n(df, groupby, sortby, n=10):
    ''' get the top n groups from a DataFrame
        :param df:      Pandas DataFrame for one meta-cluster
        :param groupby: column from DataFrame to create groups before aggregation
        :param sortby:  column from DataFrame to aggregate & sort by
        :param n:       number of groups to return
        :return         DataFrame containing data from top n groups'''
    top_n = df.groupby(
        groupby
    ).sum(
        numeric_only=True
    ).sort_values(
        by=sortby,
        ascending=False
    ).index.values[:n]

    return df[df[groupby].isin(top_n)]


### Location data related functions
@st.cache
def get_center_scale(lat, lon):
    ''' get centering and scale parameters for map display
        :param lat: list of latitudes
        :param lon: list of longitudes
        :return:    center (midpoint) and scaling '''
    midpoint = lambda lst: (max(lst) + min(lst)) / 2

    scale = lambda lst, const: const*2 / (max(lst) - min(lst)) if max(lst) - min(lst) else 100

    center = midpoint(lon), midpoint(lat)

    scale_lat = scale(lat, 90)
    scale_lon = scale(lon, 180)

    return center, min(scale_lat, scale_lon) * 100


@st.cache
def gen_locations(df):
    ''' generate latitude and longitude coordinates given city_id
        :param df:  Pandas DataFrame with column "city_id" to get coordinates from
        :return:    DataFrame with latitude and longitude data '''
    cities_df = read_csv('~/grad_projects/data/aht_data/metadata/cities.csv',
        keep_cols=['id', 'xcoord', 'ycoord'],
        rename_cols={'xcoord': 'lat', 'ycoord': 'lon'})

    return pd.merge(df, cities_df, left_on='city_id', right_on='id', sort=False)


@st.cache
def prettify_location(city, country):
    ''' make pretty location string based on city, country
        :param city:    city_id as specified by Marinus
        :param country: country_id as specified by Marinus
        :return:        string of format {state}, {country} '''
    cities_df = read_csv('~/grad_projects/data/aht_data/metadata/cities.csv')
    countries_df = read_csv('~/grad_projects/data/aht_data/metadata/countries.csv')

    country_str = countries_df[countries_df.id == country].code.values[0]
    city_str = cities_df[cities_df.id == city].name.values[0]
    return ', '.join([city_str, country_str])


@st.cache
def aggregate_locations(df):
    ''' get location counts from DataFrame
        :param df:  Pandas DataFrame with column "location"
        :return     DataFrame with location count data '''

    return df.groupby(
        ['location'],
        as_index=False
    ).agg({
         'ad_id': 'count',
         'lat': 'mean',
         'lon': 'mean'
    }).rename(
        columns={'ad_id': 'count'}
    )


### Date related
@st.cache
def extract_field_dates(df, col_name, date_col):
    ''' extract values from Pandas DataFrame with date, where one row represents multiple values
        :param df:          Pandas DataFrame to get data from 
        :param col_name:    column from DataFrame with values to extract
        :param date_col:    column from DataFrame with time data
        :return:            DataFrame with each value's count, by day '''

    df = df.dropna()
    if not len(df):
        return pd.DataFrame(columns=['metadata', 'day_posted', 'count', 'type'])

    get_data = lambda row: [(val, row[date_col]) for val in str(row[col_name]).split(';')]
    concat_reduce = lambda data: functools.reduce(lambda x, y: x + y, data)

    # expand fields that have lists, so each is a row in df
    meta_df = pd.DataFrame(concat_reduce(df.apply(get_data, axis=1)), columns=df.columns)
    # aggregate by count
    meta_df = meta_df.groupby([col_name, date_col], as_index=False).size()
    # prettify df columns
    meta_df['type'] = col_name
    meta_df = meta_df.rename(columns={'size': 'count', col_name: 'metadata'})
    return meta_df

    #return pd.DataFrame(np.concatenate(df.apply(get_data, axis=1)), columns=df.columns)


#@st.cache#(show_spinner=False)
def cluster_feature_extract(df, cluster_label='LSH label', date_col='days', loc_col='city_id'):
    ''' extract important time-based features for a particular cluster
        :param df:          Pandas DataFrame representing one meta-cluster
        :param date_col:    column from DataFrame representing time data
        :param loc_col:     column from DataFrame representing location (city id)
        :return tuple of DataFrames (1) by cluster, (2) for entire meta-cluster, (3) for metadata'''
    def total(series):
        return len(extract_field(series))

    agg_dict = {name: total for name in ('ad_id', 'city_id', 'email', 'image_id')}
    rename_dict = {
        'ad_id': '# ads',
        'city_id': '# locations',
        'email': '# emails',
        'image_id': '# images',
        'LSH label': '# clusters',
        'social': '# social media tags',
    }

    by_cluster_df = df.groupby(
        [date_col, cluster_label],
        as_index=False,
        sort=False
    ).agg(
        agg_dict
    ).rename(
        columns=rename_dict
    )

    split_prefix = lambda prefix, cell: [prefix+s for s in str(cell).split(';')]

    prefix_dict = {
        'email': '',
        'image_id': 'img-',
        'social': '@',
        'phone': ''
    }

    dfs = []
    for metadata in ('email', 'image_id', 'social', 'phone'):
        subdf = df.copy()[[metadata, 'days', 'LSH label']].dropna()
        subdf['type'] = metadata
        subdf[metadata] = subdf[metadata].apply(lambda x: split_prefix(prefix_dict[metadata], x))
        subdf = subdf.rename(columns={metadata: 'metadata'})
        #st.write(subdf, metadata in subdf)
        dfs.append(subdf.explode('metadata'))

    metadata_dict = pd.concat(dfs).groupby(
        [date_col, 'metadata', 'type'],
        as_index=False,
        sort=False
    )['LSH label'].count().reset_index().rename(
        columns={'LSH label': 'count'}
    )

    return by_cluster_df, metadata_dict


### Graph related utils
#@st.cache#(show_spinner=False)
def construct_metaclusters(filename, df, cols, cluster_label='LSH label'):
    ''' construct metadata graph from dataframe already split into clusters
    :param df:              pandas dataframe containing ad info
    :param cols:            subset of @df.columns to link clusters by
    :param cluster_label:   column from @df.columns containing cluster label 
    :return                 nx graph, where each connected component is a meta-cluster '''

    pkl_filename = 'pkl_files/{}.pkl'.format(filename)
    if os.path.exists(pkl_filename):
        return pkl.load(open(pkl_filename, 'rb'))
    
    metadata_dict = defaultdict(list)
    metadata_graph = nx.Graph()
    

    for cluster_id, cluster_df in df.groupby(cluster_label):
        if cluster_id == -1:
            continue
        metadata_graph.add_node(cluster_id, num_ads=len(cluster_df))
        
        for name in cols:
            metadata_graph.nodes[cluster_id][name] = extract_field(cluster_df[name])
            
            for elem in metadata_graph.nodes[cluster_id][name]:
                edges = [(cluster_id, node) for node in metadata_dict[elem]]
                metadata_graph.add_edges_from(edges, type=name)
                metadata_dict[elem].append(cluster_id)
                    
    pkl.dump(metadata_graph, open(pkl_filename, 'wb'))
    return metadata_graph


#@st.cache(hash_funcs={types.GeneratorType: id})#, show_spinner=False)
def gen_ccs(graph):
    ''' return generator for connected components, sorted by size
        :param graph:   nx Graph 
        :return         generator of connected components '''

    #components = sorted(nx.connected_components(graph), reverse=True, key=len)
    components = nx.connected_components(graph)
    for component in components:
        print('# clusters', len(component))
        yield component


### Text annotation utils
@st.cache
def get_all_template_text(directory):
    ''' check a directory for all possible templates and annotate them
        :param directory:   directory to check for subdirs containing templates
        :return:            string to write with annotate_text function '''
    if directory.endswith('.pkl'):
        pickled = pkl.load(open(directory, 'rb'))
        return get_template_text(*pickled, 0)
    to_write = []
    is_first = True
    for i, folder in enumerate(os.listdir(directory)):
        result_loc = '{}/{}/text.pkl'.format(directory, folder)
        if is_first:
            is_first = False
        else:
            to_write.append('<br><br>')

        pickled = pkl.load(open(result_loc, 'rb'))
        to_write += get_template_text(*pickled, i)

    return to_write

@st.cache
def get_template_text(template, ads, i):
    ''' annotate a particular template with relevant ads as calculated from InfoShield
        :param template:    list of tokens in template
        :param ads:         list of tuples of form (type_index, token)
        :param i:           template number
        :return:            string to write with annotated_text for this particular template '''
    index_to_type = {
        -1: ('slot', '#faa'),
        0:  ('const', '#fea'),
        1:  ('sub', '#8ef'),
        2:  ('del', '#aaa'),
        3:  ('ins', '#afa'),
    }

    to_write = ['Template #{}: '.format(i), ' '.join(template)]

    for ad_index, ad in enumerate(ads):
        to_write.append('<br>Ad #{}'.format(ad_index+1))
        prev_type = None 
        for color_i, token in ad:
            curr_type, color = index_to_type[color_i]

            if curr_type == prev_type:
                prev_token = to_write[-1][0]
                to_write[-1] = ('{} {}'.format(prev_token, token), curr_type, color)
                continue

            prev_type = curr_type

            to_write.append((token, curr_type, color))

    return to_write


#def write_labels(filename, meta_cluster_label, cluster_labels, labels):
#    label_filename = '{}-meta-labels.csv'.format(filename)
#    if os.path.exists(label_filename):
#        label_df = read_csv('{}-meta-labels.csv'.format(filename))
#    else:
#        label_df = pd.DataFrame(columns=['meta_cluster_label', 'cluster_labels'] + labels.keys()))
#
#    # add new row
#    row = pd.Series([meta_cluster_label, np.array(cluster_labels)] + labels.values())
#
#    label_df = label_df.append(row)
#
#    label_df.to_csv(label_filename, index=False)
