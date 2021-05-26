import altair as alt
import datetime
import numpy as np
import nx_altair as nxa
import pandas as pd

from itertools import chain, product
from vega_datasets import data

import utils
from annotated_text import annotated_text, annotation

alt.data_transformers.enable('csv')


PERSON = (
    "M1.7 -1.7h-0.8c0.3 -0.2 0.6 -0.5 0.6 -0.9c0 -0.6 "
    "-0.4 -1 -1 -1c-0.6 0 -1 0.4 -1 1c0 0.4 0.2 0.7 0.6 "
    "0.9h-0.8c-0.4 0 -0.7 0.3 -0.7 0.6v1.9c0 0.3 0.3 0.6 "
    "0.6 0.6h0.2c0 0 0 0.1 0 0.1v1.9c0 0.3 0.2 0.6 0.3 "
    "0.6h1.3c0.2 0 0.3 -0.3 0.3 -0.6v-1.8c0 0 0 -0.1 0 "
    "-0.1h0.2c0.3 0 0.6 -0.3 0.6 -0.6v-2c0.2 -0.3 -0.1 "
    "-0.6 -0.4 -0.6z"
)


def time_series(df, col):
    ''' make bar plot of time series data
        :param df:  Pandas DataFrame including "days" column
        :param col: column of DataFrame representing data to encode
        :return:    altair bar plot '''
    return alt.Chart(df).mark_bar().encode(
        x=alt.X('days', axis=alt.Axis(grid=False)),
        y=alt.Y(col, axis=alt.Axis(grid=False)),
        color=alt.value('#9467bd')
    ).properties(
        width=650,
        height=300
    )


def graph(graph, pos):
    ''' draw graph representation meta-cluster
        :param graph:   networkx graph to draw
        :param pos:     networkx layout of graph
        :return:        altair chart displaying graph '''
    node_attributes = set(chain.from_iterable(d.keys() for *_, d in graph.nodes(data=True)))
    edge_attributes = set(chain.from_iterable(d.keys() for *_, d in graph.edges(data=True)))
    return nxa.draw_networkx(
        graph, pos=pos,
        node_color='num_ads',
        cmap='tealblues',
        edge_color='grey',
        node_tooltip=list(node_attributes),
        edge_tooltip=list(edge_attributes)
    ).properties(
            width=450,
            height=400
    ).configure_view(
            strokeWidth=0
    )


def templates(directory, df, is_infoshield):
    ''' draw annotated text
        :param directory:   directory to look for InfoShield templates in
        :return:            altair annotated text '''

    if is_infoshield:
        to_write = utils.get_all_template_text(directory)
    else:
        subdf = df.iloc[0:5]
        to_write = ['{}:<br>{}'.format(*tup) for tup in subdf[['title', 'body']].values]
        to_write = [annotation(text + '<br><br>', background_color='#ffffff', font_size='20px') for text in to_write]

    annotated_text(*to_write,
        scrolling=True,
        height=400
    )


def map(df):
    ''' generate map with ad location data
        :param df:  Pandas DataFrame with latitude, longitude, and count data
        :return:    altair map with ad counts displayed '''
    df = df[(df.lat != 1) | (df.lon != 1)]
    center, scale = utils.get_center_scale(df.lat, df.lon)

    countries = alt.topo_feature(data.world_110m.url, 'countries')
    base = alt.Chart(countries, width='container').mark_geoshape(
        fill='white',
        stroke='#DDDDDD'
    ).properties(
        width=700,
        height=400
    )

    agg_df = utils.aggregate_locations(df)

    scatter = alt.Chart(agg_df, width='container').mark_circle(
        color='#ff7f0e',
        fillOpacity=.5,
    ).encode(
        size=alt.Size('count:Q', scale=alt.Scale(range=[100, 500])),
        longitude='lon:Q',
        latitude='lat:Q',
        tooltip=['location', 'count']
    )

    return (base + scatter).project(
        'equirectangular',
        scale=scale,
        center=center
    ).configure_axis(
        labelFontSize=utils.SMALL_FONT_SIZE,
        titleFontSize=utils.BIG_FONT_SIZE,
    ).configure_legend(
        gradientLength=275,
        labelFontSize=utils.SMALL_FONT_SIZE,
        titleFontSize=utils.BIG_FONT_SIZE,
    ).configure_axisX(
        labelAlign='left'
    )


def bubble_chart(df, y, facet, tooltip):
    ''' create bubble chart 
        :param df:      Pandas DataFrame to display
        :param y:       column of DataFrame to use for bubble size
        :param facet:   column of DataFrame to create facet with
        :param tooltip: list of DataFrame columns to include in tooltip
        :return:        altair bubble chart '''
    return alt.Chart(df).mark_circle().encode(
        x=alt.X('days', axis=alt.Axis(grid=True)),
        y=alt.Y(y, axis=alt.Axis(grid=False, labels=False), title=None),
        color=alt.value('#17becf'),
        row=alt.Row(facet, title=None, header=alt.Header(labelAngle=-45)),
        tooltip=tooltip,
        size=alt.Size(y, scale=alt.Scale(range=[100, 500]))
    ).properties(
        width=450,
        height=400 / len(df)
    ).configure_facet(
        spacing=5
    ).configure_view(
        stroke=None
    )


def strip_plot(df, y, facet, tooltip, sort=None, show_labels=True, colorscheme='purplered'):
    ''' create strip plot with heatmap
        :param df:      Pandas DataFrame to display
        :param y:       column of DataFrame to use for bubble size
        :param facet:   column of DataFrame to create facet with
        :param tooltip: list of DataFrame columns to include in tooltip
        :return:        altair strip plot '''

    min_date = min(df.days) - datetime.timedelta(days=1)

    return alt.Chart(df).mark_tick(thickness=20).encode(
        x=alt.X('days:T',
            axis=alt.Axis(grid=False, tickMinStep=7),
            scale=alt.Scale(domain=[min_date, max(df.days)]
        )),
        y=alt.Y(facet,
            axis=alt.Axis(grid=False, labels=show_labels),
            title='Micro-cluster',
            sort=sort
        ),
        color=alt.Color(y, scale=alt.Scale(scheme=colorscheme, type='sqrt')),
        tooltip=tooltip,
    ).properties(
        width=650,
        height=510
    ).configure_view(
        stroke=None
    ).configure_axis(
        labelFontSize=utils.SMALL_FONT_SIZE,
        titleFontSize=utils.BIG_FONT_SIZE,
    ).configure_legend(
        gradientLength=275,
        labelFontSize=utils.SMALL_FONT_SIZE,
        titleFontSize=utils.BIG_FONT_SIZE,
    ).configure_axisX(
        labelAlign='left'
    )


def labeling_buttons(title):
    ''' create buttons for labeling
        :param title:   label name for displaying as plot title
        :return:        altair plot with 5 shapes for labeling '''
    colors = ('#33cc33', '#ace600', '#e6e600', '#ff9900', '#ff3300')
    tooltip = ('1: Unlikely', '2: Somewhat Unlikely', '3: Unsure', '4: Somewhat Likely', '5: Likely')
    data = pd.DataFrame([{'id': i, 'color': c, 'label': l} for i, (c, l) in enumerate(zip(colors, tooltip))])
    brush = alt.selection_single(nearest=True, empty='none', fields=['id'])

    return alt.Chart(data).mark_point(
        filled=True,
        size=100
    ).encode(
        x=alt.X("id:O", axis=None),
        shape=alt.ShapeValue(PERSON),
        color=alt.condition(
            alt.datum.id <= brush.id,
            alt.value('#ff3300'),
            alt.value('gray')),
        tooltip=['label']
    ).properties(
        width=400,
        height=80,
        title=title
    ).configure_view(
        strokeWidth=0
    ).add_selection(
        brush
    ).configure_title(
        fontSize=utils.SMALL_FONT_SIZE
    )


def bar_chart(data, column):
    ''' create bar charts for displaying categorical data
        :param data:    data from which to display
        :param column:  column on which to create histogram
        :return:        altair bar plot '''

    return alt.Chart(data).mark_area().encode(
         x=alt.X(column, sort='-y', axis=alt.Axis(labels=False, grid=False)),
         y=alt.Y('count():Q', axis=alt.Axis(grid=False), title='Clusters (ordered by size)'),
         tooltip=[alt.Tooltip('count()', title='Number of ads in cluster')]
    ).properties(
        width=600,
        height=400
    )


def timeline(data, date_col='day_posted:T'):
    ''' create timeline for # ads each day
        :param data:    data from which to display
        :return:        altair timeline '''

    date_s = date_col.split(':')[0]

    #TODO: figure out how to calculate # of unique locations per day

    return alt.Chart(data).transform_aggregate(
        num_ads='count()',
        groupby=[date_s]
    ).transform_filter(
        alt.datum.num_ads > 1
    ).mark_point().encode(
        x=alt.X(date_col, title='Day', axis=alt.Axis(grid=False)),
        y=alt.Y('num_ads:Q', title='Number of ads', axis=alt.Axis(grid=False)),
        #tooltip=[alt.Tooltip(date_col, title='Day'), alt.Tooltip('num_ads:Q', title='Number of ads')]
    ).properties(
        width=700,
        height=400
    )


def location_timeline(data, date_col='day_posted:T'):
    ''' create timeline for # unique locations each day
        :param data:    data from which to display
        :return:        altair timeline '''

    date_s = date_col.split(':')[0]

    return alt.Chart(data).transform_aggregate(
        num_locs='distinct(city_id)',
        groupby=[date_s]
    ).mark_point().encode(
        x=alt.X(date_col, title='Day', axis=alt.Axis(grid=False)),
        y=alt.Y('num_locs:Q', title='Number of locations', axis=alt.Axis(grid=False)),
        tooltip=[alt.Tooltip(date_col, title='Day'), alt.Tooltip('num_locs:Q', title='Number of locations')]
    ).properties(
        width=700,
        height=400
    )


def contact_bar_chart(data, col):
    ''' create bar chart for metadata information
        :param data:    data from which to display
        :param col:     column name for metadata
        :return:        altair bar chart '''

    return alt.Chart(data).mark_bar().encode(
        x=alt.X('size:Q'),
        y=alt.Y(col, sort='-x'),
        color=alt.value('#40bcc9')
    ).properties(
        width=700,
        height=400
    )


def stream_chart(df):
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection_single(
        nearest=True,
        on='mouseover',
       fields=['days'],
       empty='none'
    )

    # The basic line
    line = alt.Chart(df).mark_line(interpolate='natural').encode(
        x=alt.X('days:T', axis=alt.Axis(grid=False, labels=False, title='')),
        y=alt.Y('value:Q', axis=alt.Axis(grid=False)),
        color=alt.Color('variable:N', legend=alt.Legend(orient='top'))
    ).transform_filter(
        alt.datum.variable != '# clusters'
    )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(df).mark_point().encode(
        x='days:T',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'value:Q', alt.value(' '))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(df).mark_rule(color='gray').encode(
        x='days:T',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    c1 = alt.layer(
        line, selectors, points, rules, text
    ).properties(
        width=550,
        height=200
    )

    # The basic line
    ad_line = alt.Chart(df).mark_line(interpolate='natural').encode(
        x=alt.X('days:T', axis=alt.Axis(grid=False, tickMinStep=7)),
        y=alt.Y('value:Q', axis=alt.Axis(grid=False)),
        color='variable:N'
    ).transform_filter(
        alt.datum.variable == '# clusters'
    )

    # Draw points on the line, and highlight based on selection
    ad_points = ad_line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    ad_text = ad_line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'value:Q', alt.value(' '))
    )

    # Put the five layers into a chart and bind the data
    c2 = alt.layer(
        ad_line, selectors, ad_points, rules, ad_text
    ).properties(
        width=550,
        height=100
    )

    return alt.vconcat(c1, c2,
        padding={'top': 5, 'bottom': 5, 'right': 50, 'left': 5}
    ).configure_axis(
        labelFontSize=utils.SMALL_FONT_SIZE,
        titleFontSize=utils.BIG_FONT_SIZE
    ).configure_legend(
        gradientLength=275,
        labelFontSize=utils.SMALL_FONT_SIZE,
        titleFontSize=utils.BIG_FONT_SIZE,
    ).configure_axisX(
        labelAlign='left'
    )
    