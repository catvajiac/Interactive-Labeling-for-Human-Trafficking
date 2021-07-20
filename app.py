import os
from altair.vegalite.v4.api import value
import streamlit as st

import draw
import utils
import SessionState
import altair as alt
import pandas as pd
alt.data_transformers.enable('csv')


def gen_page_content(state, df):
    ''' create Streamlit page
        :param state:           SessionState object storing cluster data
        :param df:              pandas DataFrame containing ad data '''

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

    # feature generation
    subdf = utils.get_subdf(df, state)
    header_stats, micro_cluster_features, timeline_features = utils.feature_extract(
        subdf)
    # stats = utils.basic_stats(subdf)

    utils.write_border(header_stats, state)
    if 'labels' not in st.session_state:
        st.session_state.labels = {}

    _, last_col = st.beta_columns([10, 1])
    with last_col:
        if st.button('Next meta-cluster', on_click=utils.write_labels, args=(filename, state.index, state.cluster, st.session_state.labels)):
            try:
                state.cluster = next(state.gen_clusters)
            except StopIteration:
                state.is_stop = True

    left_col, _, mid_col, _, right_col = st.beta_columns((1, 0.1, 1, 0.1, 1))

    # strip plot with heatmap
    with left_col:
        st.header('**# Ads over time**: one row is one micro-cluster')

        top_n_params, chart_params = utils.BY_CLUSTER_PARAMS
        top_df = utils.top_n(micro_cluster_features, **top_n_params)
        st.altair_chart(draw.strip_plot(top_df, **chart_params),
                        use_container_width=True)

    # display features over time, aggregated forall clusters
    with mid_col:
        st.header('**Metadata over time** of meta-cluster')
        st.altair_chart(draw.stream_chart(timeline_features),
                        use_container_width=True)

    # show map of ad locations
    with right_col:
        st.header('**Geographical spread of ads**: select a range of dates')
        date_range = pd.date_range(min(subdf.days), max(
            subdf.days)).strftime(utils.DATE_FORMAT)
        dates = st.select_slider(
            '', options=date_range, value=(date_range[0], date_range[-1]))
        st.write(
            draw.map(subdf[['ad_id', 'lat', 'lon', 'location', 'days']], dates))

    left_col, _, right_col = st.beta_columns((4, 0.1, 1))

    # template / ad text visualization
    with left_col:
        st.header('**Ad text** organized by micro-cluster')
        is_infoshield = True
        label = subdf['LSH label'].value_counts().idxmax()
        label = 6
        start_path = '../InfoShield/results/{}'.format(label)
        if not os.path.exists(start_path):
            st.warning(
                'We cannot find InfoShield results for this data, so only the ad text is displayed.')
            is_infoshield = False
        draw.templates(start_path, df, is_infoshield)

    # labeling table
    labels = []
    classes = ('Trafficking', 'Spam', 'Scam', 'Massage parlor', 'Benign')
    options = ('1: Very unlikely', '2: Unlikely',
               '3: Unsure', '4: Likely', '5: Very likely')
    for index, cluster_type in enumerate(classes):
        right_col.write(
            '<p class="label_button">{}</p>'.format(cluster_type), unsafe_allow_html=True)
        labels.append(
            right_col.select_slider('',  options, key=str(index)))

    st.session_state.labels = {class_: int(label.split(
        ':')[0]) for class_, label in zip(classes, labels)}


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

file_path = '../InfoShield/data/all_massage_LSH_labels-sample.csv'

with st.spinner('Processing data...'):
    filename = file_path
    columns = ['phone', 'email', 'social', 'image_id']

    df = utils.read_csv(filename)

    if state.is_first:
        graph = utils.construct_metaclusters(
            utils.filename_stub(filename), df, columns)
        state.gen_clusters = utils.gen_ccs(graph)

page_opts = ('Landing page', 'Labeling page')

clean_df = utils.pre_process_df(df, filename)
gen_page_content(state, clean_df)
