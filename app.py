import os, sys
import streamlit as st

import draw, utils
import SessionState
import altair as alt
import pandas as pd
alt.data_transformers.enable('csv')

def gen_landing_page_content(state, df):
    ''' create landing page
        :param state:           SessionState object storing cluster data
        :param df:              pandas DataFrame containing ad data '''

    _, col, _ = st.beta_columns([1, 2, 1])

    with col:
        st.title('Welcome to the HT labeling app!')
        st.markdown('Our purpose is to *gather high-quality labeled data* from domain experts, so we can use it to better identify the organized activities in escort ads.')

        with st.beta_expander('What do each of the labels mean? (Click here)'):
            st.markdown('''
            **1. Human Trafficking (HT):** HT ads have real victims that are in immediate danger.
            While there is much variation in what organized HT looks like, some common features of HT rings may include:
            * Many near-identical ads posted with multiple victim's names & information
            * Often a relatively small geographical spread, unless it's a national HT ring
            * A relatively large date spread -- not all ads are being posted on the same day

            **2. Spam:** Spam ads are often computer-generated and advertise escorts that don't actually exist. Some features of spam may include:
            * Many near-identical ads that look like they might be script generated, possibly advertising multiple names
            * A large geographical spread, e.g. all over the United States, or global
            * A very small date spread, having almost all ads posted in one day
            * The same person is advertised in multiple far-away locations within one day
            * Fake contact information, i.e. emails that have randomly-generated numbers at the end, or invalid phone numbers

            **3. Scam:** Like spam ads, scam ads don't usually advertise real victims. Instead, the user is trying to get money from the customer without providing any service.
            Some features of scam might include:
            * Near-identical ads
            * Location and geographical spread might vary
            * Real contact information, i.e. valid phone numbers and emails, so they can contact potential customers
            * Any mention of pre-payment / a down payment before showing up to perform sexual services

            **4. Massage Parlor:** Parlors tend to advertise real workers. They may or may not contain human trafficking activity.
            Some features of massage parlors might include:
            * Mentioning the business name or exact location
            * Talking about massages, often with happy endings
            * Generally one location, unless there is a chain of massage parlors (which can be suspicious)
            * A relatively large date spread -- not all ads are being posted on the same day

            **5. Benign:** Not an alarming organized activity, such as a hotline for trafficking victims, or an individual escort worker.
            Most likely, if a meta-cluster does not fall under any of the previous activities, it's benign.
            ''')

        st.markdown('''This dataset contains **{}** potentially suspicious meta-clusters of varying sizes.
        Here's what the distribution of cluster sizes look like...'''.format(len(df['LSH label'].unique())))
        st.altair_chart(draw.bar_chart(df[['ad_id', 'LSH label']], 'LSH label:N'), use_container_width=True)

        lens = [len(subdf) for _, subdf in df.groupby('LSH label')]
        df_stats = {
            'total': len(df),
            'min': min(lens),
            'max': max(lens),
            'avg': round(sum(lens) / len(lens))
        }

        st.markdown('''We see there are **{total}** ads posted total. On average, **{avg}** ads are posted per day, but
        this can vary widely, ranging from **{min}** to **{max}** ads per day. Below, we show the number of ads posted
        each day over the lifetime of this dataset.'''.format(**df_stats))
        st.altair_chart(draw.timeline(df[['ad_id', 'day_posted']]), use_container_width=True)

        st.markdown('''Human trafficking is a global issue. In this dataset, we see ads posted in **{}** unique cities and
        **{}** countries. Below, we show the number of locations posted over time in this dataset, as well as the posting
        behavior over time of ads in the top 10 locations.'''.format(len(df.city_id.unique()), len(df.country_id.unique())))
        top_locs = df.groupby('location').size().to_frame(name='size').reset_index().nlargest(10, 'size')
        subdf = df.copy()[['location', 'days']].dropna()
        subdf = subdf[subdf.location.isin(top_locs.location)]
        subdf = subdf.groupby(['location', 'days']).size().to_frame(name='size').reset_index()

    _, col1, col2, _ = st.beta_columns([1, 1.5, 1.5, 1])
    col1.altair_chart(
        draw.location_timeline(df[['ad_id', 'city_id', 'day_posted']]), use_container_width=True
    )

    col2.altair_chart(
        draw.strip_plot(subdf, 'size:Q', 'location', ['location', 'days', 'size'], colorscheme='tealblues'),
        use_container_width=True
    )

    _, col, _ = st.beta_columns([1, 2, 1])
    with col:
        metadata_stats = {name: len(set(utils.extract_field(df[name]))) for name in ('email', 'social', 'phone', 'image_id')}

        st.markdown('''Contact information can also give us some insight into these clusters. In this dataset, we have
        **{email}** unique email addresses, **{phone}** phone numbers, **{social}** social media tags, and **{image_id}**
        images. Below, we show the number of ads using the top 10 contact information in this dataset on the left,
        as well as the the number of ads using each contact information per day, on the right.'''
        .format(**metadata_stats))

    _, col1, col2, _ = st.beta_columns([1, 1.5, 1.5, 1])
    for metadata in ('email', 'social', 'phone', 'image_id'):
        meta_df = pd.DataFrame(utils.extract_field(df[metadata]), columns=[metadata])
        meta_df = meta_df.groupby(metadata).size().to_frame(name='size').reset_index().nlargest(10, 'size')

        subdf = df.copy()[[metadata, 'days', 'LSH label']].dropna()
        subdf[metadata] = subdf[metadata].apply(lambda x: str(x).split(';'))
        subdf = subdf.explode(metadata)
        subdf = subdf[subdf[metadata].isin(meta_df[metadata])]
        subdf = subdf.groupby([metadata, 'days']).size().to_frame(name='size').reset_index()

        with col1:
            st.altair_chart(draw.contact_bar_chart(meta_df, metadata), use_container_width=True)
        with col2:
            st.altair_chart(
                draw.strip_plot(subdf, 'size:Q', metadata, [], sort=meta_df[metadata].values, show_labels=False, colorscheme='tealblues'),
                use_container_width=True
            )

    _, col, _ = st.beta_columns([1, 2, 1])
    col.markdown('''Now that we have a sense for what this dataset as a whole looks like, the next step is to visually inspect each cluster
        to see which ones are most suspicious of human trafficking and other organized behaviors. You can use the navigation bar on the left
        to go to the labeling page, where you'll see each cluster one-by-one and can label them accordingly.''')



def gen_page_content(state, df):
    ''' create Streamlit page 
        :param state:           SessionState object storing cluster data
        :param df:              pandas DataFrame containing ad data '''

    st.markdown('''<style>
    p { font-size: 20px; }
    </style>''', unsafe_allow_html=True)
    first_col, last_col = st.beta_columns([4, 1])
    with last_col:
        st.write('  ')
        if st.button('View next meta-cluster'):
            try:
                state.cluster = next(state.gen_clusters)
                #utils.write_labels(state)
            except StopIteration:
                state.is_stop = True

    # feature generation
    subdf = utils.get_subdf(df, state)
    cluster_features, metadata_features = utils.cluster_feature_extract(subdf)
    stats = utils.basic_stats(subdf[columns + ['LSH label']], columns)

    with first_col:
        st.title('Suspicious Meta-Cluster #{}'.format(state.index+1))
        st.subheader('This cluster has: ' + utils.pretty_basic_stats(stats))

    # on first iteration, before button press
    if state.is_first:
        state.cluster = next(state.gen_clusters)
        state.index += 1
        state.is_first = False

    # if we've processed all clusters, we show a static end page
    if state.is_stop:
        st.header("You've finished all examples from this dataset. Thank you!")
        st.balloons()
        return

    left_col, _, mid_col, _, right_col = st.beta_columns((1, 0.1, 1, 0.1, 1))

    # strip plot with heatmap
    with left_col:
        st.header('Activity over time per micro-cluster.')
        #data_view = st.selectbox('Which view would you like to see?',\
        #     ['Meta-Cluster activity: # ads per top micro-clusters per day', 'Metadata usage: # clusters using top metadata per day'])

        top_n_params, chart_params = utils.BY_CLUSTER_PARAMS# if data_view.startswith('Meta-Cluster') \
        #    else utils.BY_METADATA_PARAMS
        plot_df = cluster_features #if data_view.startswith('Meta-Cluster') else metadata_features

        top_df = utils.top_n(plot_df, **top_n_params)
        st.altair_chart(draw.strip_plot(top_df, **chart_params), use_container_width=True)

    # display features over time, aggregated forall clusters
    with mid_col:
        st.header('Meta-cluster timeline of activity and metadata usage.')
        feature_cols = [f for f in cluster_features if f not in ('days')]
        features = cluster_features.groupby('days', as_index=False).agg('sum')
        melt = pd.melt(features, id_vars=['days'], value_vars=feature_cols)
        st.altair_chart(draw.stream_chart(melt), use_container_width=True)

    # show map of ad locations
    with right_col:
        st.header('Geographical spread of advertisements.')
        st.write(draw.map(subdf[['ad_id', 'lat', 'lon', 'location']]))


    left_col, _, right_col = st.beta_columns((4, 0.1, 1))

    # template / ad text visualization
    with left_col:
        st.header('Ad text, organized by micro-cluster')
        is_infoshield = True
        label = subdf['LSH label'].value_counts().idxmax()
        start_path = '../InfoShield/results/{}'.format(label)
        if not os.path.exists(start_path):
            st.warning('We cannot find InfoShield results for this data, so only the ad text is displayed.')
            is_infoshield = False
        draw.templates(start_path, df, is_infoshield)

    # labeling table
    with right_col:
        st.header('Labeling: On a scale of 1 (very unlikely) to 5 (very likely), how likely is this to be...')
        for cluster_type in ('Trafficking', 'Spam', 'Scam', 'Massage parlor', 'Benign'):
            st.write(draw.labeling_buttons(cluster_type))


# Generate content for app
st.set_page_config(layout='wide', page_title='Meta-Clustering Classification')
state_params = {
    'is_first': True,
    'index': 0,
    'cluster': set(),
    'is_stop': False,
    'gen_clusters': None
}
state = SessionState.get(**state_params)

default_file_path = './data/synthetic_data.csv'
#file_path = st.text_input("Please specify the path of input file")
file_path = ''

try:
    if not os.path.exists(file_path):
        if file_path: # only show warning if user already tried to input file path
            st.warning("The file does not exist, displaying default dataset.")
            st.warning("If you would like to use your own dataset, please specify the path again.")
        file_path = default_file_path
except:
    st.warning("Path not correct. Please spcecify a path again.")
    file_path = default_file_path

with st.spinner('Processing data...'):
    filename = file_path
    columns = ['phone', 'email', 'social', 'image_id']

    df = utils.read_csv(filename)

    if state.is_first:
        graph = utils.construct_metaclusters(
            utils.filename_stub(filename), df, columns)
        state.gen_clusters = utils.gen_ccs(graph)

st.sidebar.title('Navigation')

page_opts = ('Landing page', 'Labeling page')
choose_page = st.sidebar.radio('Go to', page_opts)

clean_df = utils.pre_process_df(df, filename)

if choose_page == page_opts[0]:
    gen_landing_page_content(state, clean_df)
else:
    gen_page_content(state, clean_df)
