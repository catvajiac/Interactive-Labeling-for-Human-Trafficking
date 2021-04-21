import os, sys
import streamlit as st

import draw, utils
import SessionState


def gen_page_content(state, df):
    ''' create Streamlit page 
        :param state:           SessionState object storing cluster data
        :param df:              pandas DataFrame containing ad data '''
    first_col, _, _, _, last_col = st.beta_columns(5)
    with last_col:
        if st.button('View next cluster'):
            try:
                state.index, state.cluster = next(state.gen_clusters)
                #utils.write_labels(state)
            except StopIteration:
                state.is_stop = True

    with first_col:
        st.title('Suspicious Meta-Cluster #{}'.format(state.index+1))

    # on first iteration, before button press
    if state.is_first:
        next(state.gen_clusters) # TODO: change once scalable
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
    cluster_features, features, metadata_features = utils.cluster_feature_extract(subdf)

    left_col, right_col = st.beta_columns((1.25, 3))

    # strip plot with heatmap
    with left_col:
        st.write(utils.BUTTON_STYLE, unsafe_allow_html=True) # allows side-by-side button opts
        radio_val = st.radio('Which view would you like to see?', ['By cluster', 'By metadata'])

        top_n_params, chart_params = utils.BY_CLUSTER_PARAMS if radio_val == 'By cluster' \
            else utils.BY_METADATA_PARAMS
        plot_df = cluster_features if radio_val == 'By cluster' else metadata_features

        top_df = utils.top_n(plot_df, **top_n_params)
        st.write(draw.strip_plot(top_df, **chart_params))

    # template / ad text visualization
    with right_col:
        label = subdf['LSH label'].value_counts().idxmax()
        start_path = '../InfoShield/results/{}'.format(label)
        start_path = './data/example.pkl'
        if not os.path.exists(start_path):
            start_path = './data/example.pkl'
        draw.templates(start_path)

    # hacky way to get padding between columns
    left_col, _, mid_col, _, right_col = st.beta_columns((1, 0.1, 1.5, 0.1, 2))

    # meta-cluster stats table
    with left_col:
        st.subheader('Meta-Cluster Stats')
        st.table(utils.basic_stats(subdf, columns))

    # display features over time, aggregated forall clusters
    with mid_col:
        select_feature = st.selectbox('Choose a feature to look at over time', [f for f in features if f != 'days'])
        if select_feature:
            st.write(draw.time_series(features, select_feature))

    # show map of ad locations
    with right_col:
        st.write(draw.map(subdf))

    # Number input boxes take up the whole column space -- this makes them shorter
    st.subheader('Labeling: On a scale of 1 (very unlikely) to 5 (very likely), how likely is this to be...')
    label_cols = st.beta_columns(5)

    for col, cluster_type in zip(label_cols, ('Trafficking', 'Spam', 'Scam', 'Drug dealer', 'Other')):
        col.write(draw.labeling_buttons(cluster_type))


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
default_file_path = '../data/aht_data/aggregate_data/aggregate_LSH_labels.csv'
file_path = st.text_input("Please specify the path of input file")

try:
    if not os.path.exists(file_path):
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

gen_page_content(state, df)
