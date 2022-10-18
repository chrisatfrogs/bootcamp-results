import json


import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.transforms as mtransforms
import numpy as np
from scipy.stats import t
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from annotated_text import annotated_text, util

from constants import COUNT_COLUMNS, LIKERT_COLUMNS, COUNT_MAPPER, LIKERT_MAPPER, INV_COUNT_MAPPER, INV_LIKERT_MAPPER, ALL_PERCENT_COLUMNS, COMMENTS

@st.cache
def load_data():
    with open('bootcamp_df.json') as f:
        data = json.load(f)
    return data


def get_data(user: str) -> pd.DataFrame:
    data = load_data()
    base_df = pd.DataFrame.from_dict(data[user])
    df = pd.DataFrame(index = ALL_PERCENT_COLUMNS, columns = ['user_mean', 'm_mean', 'p_value'])
    for ix, row in base_df.iterrows():
        for col in ALL_PERCENT_COLUMNS:
            if ix in df.columns:
                df.loc[col, ix] = row[col]
    df.reset_index(inplace=True)
    df['index'] = df['index'].str.replace('_percent', '')
    df.set_index('index', inplace=True)
    return df

def style_df(df: pd.DataFrame) -> pd.DataFrame:
    style_df = df.copy()
    style_df.loc[:, :] = ''
    style_df['p_value'] = df['p_value'].apply(lambda x: f'background-color: #2f5e00; color: #ffffff' if x <= 0.05 else '')
    return style_df


def get_chart_data(df: pd.DataFrame) -> list:
    header = list(df.index)
    data = []
    data.append(header)
    for col in df.columns[:-1]:
        if col == 'user_mean':
            data.append(('Your ratings', list(df[col])))
        elif col == 'm_mean':
            data.append(("Ella's ratings", list(df[col])))
    return data

def generate_plotly_chart(data: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=data['user_mean'],
        theta=data.index,
        fill='toself',
        name='Your ratings'
    ))
    fig.add_trace(go.Scatterpolar(
        r=data['m_mean'],
        theta=data.index,
        fill='toself',
        name="Ella's ratings"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )
    return fig



def radar_factory(num_vars: int, frame: str = 'circle'):
    """
    This function creates a RadarAxes projection with 'num_vars' axes and registers it.


    :param int num_vars: number of axes for the radar chart
    :param str frame: shape of frame surrounding axes {'circle' | 'polygon'}
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(mtransforms.Transform):
        """
        The base polar transform. This handles projection *theta* and
        *r* into Cartesian coordinate space *x* and *y*, but does not
        perform the ultimate affine transformation into the correct
        position.

        This is copied from Matplotlib version 3.2.2 since in 3.3.0
        the grid lines are using a different interpolation method.
        """
        input_dims = output_dims = 2

        def __init__(self, axis=None, use_rmin=True,
                     _apply_theta_transforms=True):
            mtransforms.Transform.__init__(self)
            self._axis = axis
            self._use_rmin = use_rmin
            self._apply_theta_transforms = _apply_theta_transforms

        def transform_non_affine(self, tr):
            # docstring inherited
            t, r = np.transpose(tr)
            # PolarAxes does not use the theta transforms here, but apply them for
            # backwards-compatibility if not being used by it.
            if self._apply_theta_transforms and self._axis is not None:
                t *= self._axis.get_theta_direction()
                t += self._axis.get_theta_offset()
            if self._use_rmin and self._axis is not None:
                r = (r - self._axis.get_rorigin()) * self._axis.get_rsign()
            r = np.where(r >= 0, r, np.nan)
            return np.column_stack([r * np.cos(t), r * np.sin(t)])

        def transform_path_non_affine(self, path):
            # docstring inherited
            vertices = path.vertices
            if len(vertices) == 2 and vertices[0, 0] == vertices[1, 0]:
                return mpath.Path(self.transform(vertices), path.codes)
            ipath = path.interpolated(path._interpolation_steps)
            return mpath.Path(self.transform(ipath.vertices), ipath.codes)

        def inverted(self):
            # docstring inherited
            return PolarAxes.InvertedPolarTransform(self._axis, self._use_rmin, self._apply_theta_transforms)

    class RadarAxes(PolarAxes):
        """
        Axes for the radar plot. The layout can either be a circle or a polygon with 'num_vars' vertices.
        """
        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5 in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer, *args, **kwargs):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer, *args, **kwargs)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at (0, 0) but we want a polygon
                # of radius 0.5 centered at (0.5, 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)

                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    RadarAxes.PolarTransform = RadarTransform
    register_projection(RadarAxes)
    return theta

def generate_radar_chart(num_vars: int, data: list):
    # plt.style.use('dark_background')
    theta = radar_factory(num_vars, frame='polygon')
    spoke_labels = data.pop(0)
    fig = plt.figure(figsize=(3, 3))
    fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.75, hspace=1, top=0.85, bottom=0.05)

    colors = ['b', 'r']
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    
    for (title, case_data) in data:
        if title == "Your ratings":
            color = colors[0]
        else:
            color = colors[1]
        ax.plot(theta, case_data, color=color)
        ax.fill(theta, case_data, facecolor=color, alpha=0.25, label='_nolegend_')
    ax.set_varlabels(spoke_labels)
    labels = ('Yours', 'Ella')
    ax.legend(labels, loc=(0.9, .95), labelspacing=0.2, fontsize='medium')
    fig.text(0.5, 0.965, 'Average ratings for the English bootcamp texts',
             horizontalalignment='center', color='black', weight='bold',
             size='large')
    return fig





# Set the page layout to wide
st.set_page_config(layout="wide", page_title="Bootcamp results")

def main():
    st.title('Bootcamp results')
    main_data = load_data()
    users = list(main_data.keys())
    user = st.sidebar.selectbox('Select a user', users)
    df = get_data(user)
    chart_data = get_chart_data(df)
    col1, col2 = st.columns(2)
    with col1:
        # st.plotly_chart(generate_plotly_chart(df))
        st.pyplot(generate_radar_chart(11, chart_data))
    with col2:
        st.subheader('Tabulated results')
        # st.plotly_chart(generate_plotly_chart(df))
        st.dataframe(df.style.apply(style_df, axis = None), use_container_width=True, height = 420)
    
    st.markdown('<br />', unsafe_allow_html=True)
    if user == 'anna.baumann':
        st.markdown('<br />', unsafe_allow_html=True)
        st.markdown('### Comments')
        st.markdown('''
        There are no *obviously* significant differences between your ratings and Ella's. However, the ratings for the criterion **overall_quality** are close to the threshold value (p_value = 0.06). While this difference is in this context not considered statistically significant, it is still worth analyzing. 
        ''')
        col3, col4 = st.columns(2)
        st.markdown('<br />', unsafe_allow_html=True)
        with st.expander('Text analysis'):
            with col3:
                st.markdown('##### Input text')
                st.write('''
                Russia will limit exports of nitrogen fertilisers and complex nitrogen-containing fertilisers for six months to try to curb any further increase in food prices amid higher gas prices, Prime Minister Mikhail Mishustin said on Wednesday. "The increase in natural gas prices has had a negative impact on the global markets," Mishustin told a government meeting. "This could lead to higher prices and shortages of nitrogen fertilisers, in the production of which the price of gas occupies a significant part, and, as a consequence, cause an increase in the cost of food products. To prevent this, the government is imposing temporary restrictions on the exports," he added. The quota for exports of nitrogen fertilisers will be set at 5.9 million tonnes, and for complex nitrogen-containing fertilisers at 5.35 million tonnes, he said.
                ''')
            with col4:
                st.markdown('##### Output text')
                st.write('''
                Prime Minister Mikhail Mishustin said on Wednesday that Russia would limit exports of nitrogen fertilisers and complex nitrogen-containing fertilisers for the six months to try to curb any further increase in food prices due to higher gas prices. "The increase in natural gas prices has had a negative impact on the global markets," Mishustin told a government meeting. "This could lead to higher prices and shortages of nitrogen fertilisers, in the production of which the price of gas occupies a significant part, and, as a consequence, cause an increase in the cost of food products. To prevent this, the government is imposing temporary restrictions on the exports," he added. He said that the quota for nitrogen fertilisers would be set at 5.9 million tonnes and complex nitrogen-containing fertilisers at 5.35 tonnes.
                ''')
            st.markdown('<br />', unsafe_allow_html=True)
            annotated_text(
                'Prime Minister Mikhail Mishustin said on Wednesday that Russia would limit exports of nitrogen fertilisers and complex nitrogen-containing fertilisers for ',
                ('the six months', 'GR'),
                ' to try to curb any further increase in food prices ',
                ('due to higher gas prices. ', 'LC'),
                '"The increase in natural gas prices has had a negative impact on the global markets," Mishustin told a government meeting. "This could lead to higher prices and shortages of nitrogen fertilisers, in the production of which the price of gas occupies a significant part, and, as a consequence, cause an increase in the cost of food products. To prevent this, the government is imposing temporary restrictions on the exports," he added.',
                ('He said that the quota for nitrogen fertilisers', 'EI'),
                'would be set at 5.9 million tonnes and complex nitrogen-containing fertilisers at ',
                ('5.35 tonnes.', 'EI')
            )
            st.markdown('<br />', unsafe_allow_html=True)
            annotated_text(
                ('Prime Minister Mikhail Mishustin said on Wednesday that Russia would limit exports of nitrogen fertilisers and complex nitrogen-containing fertilisers for the six months to try to curb any further increase in food prices due to higher gas prices.', 'changed'),
                ('"The increase in natural gas prices has had a negative impact on the global markets," Mishustin told a government meeting.', 'unchanged'),
                ('"This could lead to higher prices and shortages of nitrogen fertilisers, in the production of which the price of gas occupies a significant part, and, as a consequence, cause an increase in the cost of food products.', 'unchanged'),
                ('"To prevent this, the government is imposing temporary restrictions on the exports," he added.', 'unchanged'),
                ('He said that the quota for nitrogen fertilisers would be set at 5.9 million tonnes and complex nitrogen-containing fertilisers at 5.35 tonnes.', 'changed')
            )
            st.markdown('<br />', unsafe_allow_html=True)

    elif user == 'theresa.herrmann':
        st.markdown('<br />', unsafe_allow_html=True)
        st.markdown('### Comments')
        st.markdown('''
        Significant differences can be found in the ratings for the criteria **grammar**, **factual_correctness** and **lexical_correctness**. 


        ''')
        col5, col6 = st.columns(2)
        st.markdown('<br />', unsafe_allow_html=True)
        with col5:
            st.markdown('##### Input text')
            st.write('''
            Fox Corp chief executive Lachlan Murdoch will return to the United States from a months-long Australian sojourn in early September, a spokesman told Reuters on Wednesday. Murdoch, who also co-chairs publisher News Corp with his father Rupert Murdoch, moved with his family to Sydney last month, joining a host of high profile people who have opted to ride out the coronavirus pandemic in relatively unaffected Australia. Spokesman John Connolly said Murdoch, who is also Fox's executive chairman, would return to Los Angeles in early September to reopen the company's operational head office there "and will go back to working and living in the U.S." Fox last month told its 9,000 staff to return to their U.S. offices from Sept. 7, after working at from home for several months. "As with most CEOs of large companies, he's looking forward to moving from working from home to working from the office," Connolly said, adding that Murdoch would commute regularly to Sydney where his immediate family will remain. While he is in Sydney, Murdoch will work midnight to morning local time, a source familiar with the plans said. Australia and the United States have had vastly different pandemic experiences. While an average of around 69,500 new COVID-19 infections are being reported in the United States each day, Australia has a near zero rate. The surge in U.S. cases led Biden to urge states on March 29 to pause reopening efforts as a top health official warned of "impending doom."
            ''')
        with col6:
            st.markdown('##### Output text')
            st.write('''
           Fox Corp chief executive Lachlan Murdoch will return to the United States from early September after a months-long sojourn in Australia, a spokesman told Reuters on Wednesday. Murdoch, who co-chairs publisher News Corp with his father Rupert Murdoch, moved with his family to Sydney last month, joining a number of high profile people who have decided to ride out the coronaviruses in relatively unaffected Australia. Murdoch is the Fox's executive chairman, would return to Los Angeles in early September to reopen the company"s operational head office there, according to spokesman John Connolly. The 9,000 staff were told by Fox to return to their U.S. offices from Sept. 7, after working from home for several months. Murdoch would commute frequently to Sydney, where his family will remain, according to "As with most CEOs of large companies, he's looking forward to moving from working from home to working from the office," Connolly. A source familiar with the plans said that Murdoch would work midnight to morning local time while he was in Sydney. The United States and Australia have had different experiences with the Pandemic. In the United States, around 69,500 new COVID-19 infections are reported each day, but in Australia, there is a near zero rate. Biden urged states to pause reopening efforts on March 29 as a top health official warned of "impending doom."
            ''')
        st.markdown('<br />', unsafe_allow_html=True)
        with st.expander('Text analysis (Output text)'):
            st.markdown('###### Count criteria')
            annotated_text(
                'Fox Corp chief executive Lachlan Murdoch will return to the United States from early September after a months-long sojourn in Australia, a spokesman told Reuters on Wednesday. Murdoch, who co-chairs publisher News Corp with his father Rupert Murdoch, moved with his family to Sydney last month, joining a number of high profile people who have decided to ride out the ',
                ('coronaviruses', 'GR LC'),
                'in relatively unaffected Australia. Murdoch is ',
                ('the Fox\'s executive chairman', 'GR'), 
                'would return to Los Angeles in early September to reopen the ',
                ('company"s', 'GR'), 
                'operational head office there, according to spokesman John Connolly.',
                util.annotation('("and will go back to working and living in the U.S.")', 'EI', border="1px dashed red"),
                'The 9,000 staff were told by Fox to return to their U.S. offices from Sept. 7, after working from home for several months. Murdoch would commute frequently to Sydney, where his family will remain, according to ',
                ('"As with most CEOs of large companies, he\'s looking forward to moving from working from home to working from the office," Connolly.','GR SY'), 
                'A source familiar with the plans said that Murdoch would work midnight to morning local time while he was in Sydney. The United States and Australia have had different experiences with',
                ('the Pandemic.', 'GR'), 
                'In the United States, around 69,500 new COVID-19 infections are reported each day, but in Australia, there is a near zero rate. ',
                util.annotation('(The surge in U.S. cases led)', 'EI', border="1px dashed red"), 
                'Biden urged states to pause reopening efforts on March 29 as a top health official warned of "impending doom."'
            )
            st.markdown('<br />', unsafe_allow_html=True)
            st.markdown('###### Likert criteria')
            annotated_text(
                ('Fox Corp chief executive Lachlan Murdoch will return to the United States from early September after a months-long sojourn in Australia, a spokesman told Reuters on Wednesday. ', 'slightly changed'),
                ('Murdoch, who co-chairs publisher News Corp with his father Rupert Murdoch, moved with his family to Sydney last month, joining a number of high profile people who have decided to ride out the coronaviruses in relatively unaffected Australia. ', 'slightly changed'),
                ('Murdoch is the Fox\'s executive chairman, would return to Los Angeles in early September to reopen the company"s operational head office there, according to spokesman John Connolly. ', 'changed with errors'),
                ('The 9,000 staff were told by Fox to return to their U.S. offices from Sept. 7, after working from home for several months.', 'slightly changed'),
                ('Murdoch would commute frequently to Sydney, where his family will remain, according to "As with most CEOs of large companies, he\'s looking forward to moving from working from home to working from the office," Connolly.', 'changed with errors'),
                ('A source familiar with the plans said that Murdoch would work midnight to morning local time while he was in Sydney.', 'slightly changed'),
                ('The United States and Australia have had different experiences with the Pandemic.', 'changed with errors'),
                ('In the United States, around 69,500 new COVID-19 infections are reported each day, but in Australia, there is a near zero rate.', 'changed'),
                ('Biden urged states to pause reopening efforts on March 29 as a top health official warned of "impending doom."', 'slightly changed with errors')
            )
            st.markdown('<br />', unsafe_allow_html=True)
    elif user == 'emma.carballal-haire':
        st.markdown('<br />', unsafe_allow_html=True)
        st.markdown('### Comments')
        st.markdown('''
        Significant differences can be found in the ratings for the criteria **grammar** and **syntactic_correctness**. 


        ''')
        col5, col6 = st.columns(2)
        st.markdown('<br />', unsafe_allow_html=True)
        with col5:
            st.markdown('##### Input text')
            st.write('''
            Fox Corp chief executive Lachlan Murdoch will return to the United States from a months-long Australian sojourn in early September, a spokesman told Reuters on Wednesday. Murdoch, who also co-chairs publisher News Corp with his father Rupert Murdoch, moved with his family to Sydney last month, joining a host of high profile people who have opted to ride out the coronavirus pandemic in relatively unaffected Australia. Spokesman John Connolly said Murdoch, who is also Fox's executive chairman, would return to Los Angeles in early September to reopen the company's operational head office there "and will go back to working and living in the U.S." Fox last month told its 9,000 staff to return to their U.S. offices from Sept. 7, after working at from home for several months. "As with most CEOs of large companies, he's looking forward to moving from working from home to working from the office," Connolly said, adding that Murdoch would commute regularly to Sydney where his immediate family will remain. While he is in Sydney, Murdoch will work midnight to morning local time, a source familiar with the plans said. Australia and the United States have had vastly different pandemic experiences. While an average of around 69,500 new COVID-19 infections are being reported in the United States each day, Australia has a near zero rate. The surge in U.S. cases led Biden to urge states on March 29 to pause reopening efforts as a top health official warned of "impending doom."
            ''')
        with col6:
            st.markdown('##### Output text')
            st.write('''
           Fox Corp chief executive Lachlan Murdoch will return to the United States from early September after a months-long sojourn in Australia, a spokesman told Reuters on Wednesday. Murdoch, who co-chairs publisher News Corp with his father Rupert Murdoch, moved with his family to Sydney last month, joining a number of high profile people who have decided to ride out the coronaviruses in relatively unaffected Australia. Murdoch is the Fox's executive chairman, would return to Los Angeles in early September to reopen the company"s operational head office there, according to spokesman John Connolly. The 9,000 staff were told by Fox to return to their U.S. offices from Sept. 7, after working from home for several months. Murdoch would commute frequently to Sydney, where his family will remain, according to "As with most CEOs of large companies, he's looking forward to moving from working from home to working from the office," Connolly. A source familiar with the plans said that Murdoch would work midnight to morning local time while he was in Sydney. The United States and Australia have had different experiences with the Pandemic. In the United States, around 69,500 new COVID-19 infections are reported each day, but in Australia, there is a near zero rate. Biden urged states to pause reopening efforts on March 29 as a top health official warned of "impending doom."
            ''')
        st.markdown('<br />', unsafe_allow_html=True)
        with st.expander('Text analysis (Output text)'):
            st.markdown('###### Count criteria')
            annotated_text(
                'Fox Corp chief executive Lachlan Murdoch will return to the United States from early September after a months-long sojourn in Australia, a spokesman told Reuters on Wednesday. Murdoch, who co-chairs publisher News Corp with his father Rupert Murdoch, moved with his family to Sydney last month, joining a number of high profile people who have decided to ride out the ',
                ('coronaviruses', 'GR LC'),
                'in relatively unaffected Australia. Murdoch is ',
                ('the Fox\'s executive chairman', 'GR'), 
                'would return to Los Angeles in early September to reopen the ',
                ('company"s', 'GR'), 
                'operational head office there, according to spokesman John Connolly.',
                util.annotation('("and will go back to working and living in the U.S.")', 'EI', border="1px dashed red"),
                'The 9,000 staff were told by Fox to return to their U.S. offices from Sept. 7, after working from home for several months. Murdoch would commute frequently to Sydney, where his family will remain, according to ',
                ('"As with most CEOs of large companies, he\'s looking forward to moving from working from home to working from the office," Connolly.','GR SY'), 
                'A source familiar with the plans said that Murdoch would work midnight to morning local time while he was in Sydney. The United States and Australia have had different experiences with',
                ('the Pandemic.', 'GR'), 
                'In the United States, around 69,500 new COVID-19 infections are reported each day, but in Australia, there is a near zero rate. ',
                util.annotation('(The surge in U.S. cases led)', 'EI', border="1px dashed red"), 
                'Biden urged states to pause reopening efforts on March 29 as a top health official warned of "impending doom."'
            )
            st.markdown('<br />', unsafe_allow_html=True)
            st.markdown('###### Likert criteria')
            annotated_text(
                ('Fox Corp chief executive Lachlan Murdoch will return to the United States from early September after a months-long sojourn in Australia, a spokesman told Reuters on Wednesday. ', 'slightly changed'),
                ('Murdoch, who co-chairs publisher News Corp with his father Rupert Murdoch, moved with his family to Sydney last month, joining a number of high profile people who have decided to ride out the coronaviruses in relatively unaffected Australia. ', 'slightly changed'),
                ('Murdoch is the Fox\'s executive chairman, would return to Los Angeles in early September to reopen the company"s operational head office there, according to spokesman John Connolly. ', 'changed with errors'),
                ('The 9,000 staff were told by Fox to return to their U.S. offices from Sept. 7, after working from home for several months.', 'slightly changed'),
                ('Murdoch would commute frequently to Sydney, where his family will remain, according to "As with most CEOs of large companies, he\'s looking forward to moving from working from home to working from the office," Connolly.', 'changed with errors'),
                ('A source familiar with the plans said that Murdoch would work midnight to morning local time while he was in Sydney.', 'slightly changed'),
                ('The United States and Australia have had different experiences with the Pandemic.', 'changed with errors'),
                ('In the United States, around 69,500 new COVID-19 infections are reported each day, but in Australia, there is a near zero rate.', 'changed'),
                ('Biden urged states to pause reopening efforts on March 29 as a top health official warned of "impending doom."', 'slightly changed with errors')
            )
            st.markdown('<br />', unsafe_allow_html=True)
    elif user == 'larissa.rath':
        st.markdown('<br />', unsafe_allow_html=True)
        st.markdown('### Comments')
        st.markdown('''
        Significant differences can be found in the ratings for the criteria **factual_correctness**, **syntactic_correctness**, **content_similarity** and **overall_quality**.


        ''')
        col7, col8 = st.columns(2)
        st.markdown('<br />', unsafe_allow_html=True)
        with col7:
            st.markdown('##### Input text')
            st.write('''
            Fox Corp chief executive Lachlan Murdoch will return to the United States from a months-long Australian sojourn in early September, a spokesman told Reuters on Wednesday. Murdoch, who also co-chairs publisher News Corp with his father Rupert Murdoch, moved with his family to Sydney last month, joining a host of high profile people who have opted to ride out the coronavirus pandemic in relatively unaffected Australia. Spokesman John Connolly said Murdoch, who is also Fox's executive chairman, would return to Los Angeles in early September to reopen the company's operational head office there "and will go back to working and living in the U.S." Fox last month told its 9,000 staff to return to their U.S. offices from Sept. 7, after working at from home for several months. "As with most CEOs of large companies, he's looking forward to moving from working from home to working from the office," Connolly said, adding that Murdoch would commute regularly to Sydney where his immediate family will remain. While he is in Sydney, Murdoch will work midnight to morning local time, a source familiar with the plans said. Australia and the United States have had vastly different pandemic experiences. While an average of around 69,500 new COVID-19 infections are being reported in the United States each day, Australia has a near zero rate. The surge in U.S. cases led Biden to urge states on March 29 to pause reopening efforts as a top health official warned of "impending doom."
            ''')
        with col8:
            st.markdown('##### Output text')
            st.write('''
           Fox Corp chief executive Lachlan Murdoch will return to the United States from early September after a months-long sojourn in Australia, a spokesman told Reuters on Wednesday. Murdoch, who co-chairs publisher News Corp with his father Rupert Murdoch, moved with his family to Sydney last month, joining a number of high profile people who have decided to ride out the coronaviruses in relatively unaffected Australia. Murdoch is the Fox's executive chairman, would return to Los Angeles in early September to reopen the company"s operational head office there, according to spokesman John Connolly. The 9,000 staff were told by Fox to return to their U.S. offices from Sept. 7, after working from home for several months. Murdoch would commute frequently to Sydney, where his family will remain, according to "As with most CEOs of large companies, he's looking forward to moving from working from home to working from the office," Connolly. A source familiar with the plans said that Murdoch would work midnight to morning local time while he was in Sydney. The United States and Australia have had different experiences with the Pandemic. In the United States, around 69,500 new COVID-19 infections are reported each day, but in Australia, there is a near zero rate. Biden urged states to pause reopening efforts on March 29 as a top health official warned of "impending doom."
            ''')
        st.markdown('<br />', unsafe_allow_html=True)
        with st.expander('Text analysis (Output text)'):
            st.markdown('###### Count criteria')
            annotated_text(
                'Fox Corp chief executive Lachlan Murdoch will return to the United States from early September after a months-long sojourn in Australia, a spokesman told Reuters on Wednesday. Murdoch, who co-chairs publisher News Corp with his father Rupert Murdoch, moved with his family to Sydney last month, joining a number of high profile people who have decided to ride out the ',
                ('coronaviruses', 'GR LC'),
                'in relatively unaffected Australia. Murdoch is ',
                ('the Fox\'s executive chairman', 'GR'), 
                'would return to Los Angeles in early September to reopen the ',
                ('company"s', 'GR'), 
                'operational head office there, according to spokesman John Connolly.',
                util.annotation('("and will go back to working and living in the U.S.")', 'EI', border="1px dashed red"),
                'The 9,000 staff were told by Fox to return to their U.S. offices from Sept. 7, after working from home for several months. Murdoch would commute frequently to Sydney, where his family will remain, according to ',
                ('"As with most CEOs of large companies, he\'s looking forward to moving from working from home to working from the office," Connolly.','GR SY'), 
                'A source familiar with the plans said that Murdoch would work midnight to morning local time while he was in Sydney. The United States and Australia have had different experiences with',
                ('the Pandemic.', 'GR'), 
                'In the United States, around 69,500 new COVID-19 infections are reported each day, but in Australia, there is a near zero rate. ',
                util.annotation('(The surge in U.S. cases led)', 'EI', border="1px dashed red"), 
                'Biden urged states to pause reopening efforts on March 29 as a top health official warned of "impending doom."'
            )
            st.markdown('<br />', unsafe_allow_html=True)
            st.markdown('###### Likert criteria')
            annotated_text(
                ('Fox Corp chief executive Lachlan Murdoch will return to the United States from early September after a months-long sojourn in Australia, a spokesman told Reuters on Wednesday. ', 'slightly changed'),
                ('Murdoch, who co-chairs publisher News Corp with his father Rupert Murdoch, moved with his family to Sydney last month, joining a number of high profile people who have decided to ride out the coronaviruses in relatively unaffected Australia. ', 'slightly changed'),
                ('Murdoch is the Fox\'s executive chairman, would return to Los Angeles in early September to reopen the company"s operational head office there, according to spokesman John Connolly. ', 'changed with errors'),
                ('The 9,000 staff were told by Fox to return to their U.S. offices from Sept. 7, after working from home for several months.', 'slightly changed'),
                ('Murdoch would commute frequently to Sydney, where his family will remain, according to "As with most CEOs of large companies, he\'s looking forward to moving from working from home to working from the office," Connolly.', 'changed with errors'),
                ('A source familiar with the plans said that Murdoch would work midnight to morning local time while he was in Sydney.', 'slightly changed'),
                ('The United States and Australia have had different experiences with the Pandemic.', 'changed with errors'),
                ('In the United States, around 69,500 new COVID-19 infections are reported each day, but in Australia, there is a near zero rate.', 'changed'),
                ('Biden urged states to pause reopening efforts on March 29 as a top health official warned of "impending doom."', 'slightly changed with errors')
            )
            st.markdown('<br />', unsafe_allow_html=True)
        col9, col10 = st.columns(2)
        st.markdown('<br />', unsafe_allow_html=True)
        with col9:
            st.markdown('##### Input text')
            st.write('''
                Russia will limit exports of nitrogen fertilisers and complex nitrogen-containing fertilisers for six months to try to curb any further increase in food prices amid higher gas prices, Prime Minister Mikhail Mishustin said on Wednesday. "The increase in natural gas prices has had a negative impact on the global markets," Mishustin told a government meeting. "This could lead to higher prices and shortages of nitrogen fertilisers, in the production of which the price of gas occupies a significant part, and, as a consequence, cause an increase in the cost of food products. To prevent this, the government is imposing temporary restrictions on the exports," he added. The quota for exports of nitrogen fertilisers will be set at 5.9 million tonnes, and for complex nitrogen-containing fertilisers at 5.35 million tonnes, he said.
                ''')
        with col10:
            st.markdown('##### Output text')
            st.write('''
                Prime Minister Mikhail Mishustin said on Wednesday that Russia would limit exports of nitrogen fertilisers and complex nitrogen-containing fertilisers for the six months to try to curb any further increase in food prices due to higher gas prices. "The increase in natural gas prices has had a negative impact on the global markets," Mishustin told a government meeting. "This could lead to higher prices and shortages of nitrogen fertilisers, in the production of which the price of gas occupies a significant part, and, as a consequence, cause an increase in the cost of food products. To prevent this, the government is imposing temporary restrictions on the exports," he added. He said that the quota for nitrogen fertilisers would be set at 5.9 million tonnes and complex nitrogen-containing fertilisers at 5.35 tonnes.
                ''')
        st.markdown('<br />', unsafe_allow_html=True)
        with st.expander('Text analysis'):
            annotated_text(
                'Prime Minister Mikhail Mishustin said on Wednesday that Russia would limit exports of nitrogen fertilisers and complex nitrogen-containing fertilisers for ',
                ('the six months', 'GR'),
                ' to try to curb any further increase in food prices ',
                ('due to higher gas prices. ', 'LC'),
                '"The increase in natural gas prices has had a negative impact on the global markets," Mishustin told a government meeting. "This could lead to higher prices and shortages of nitrogen fertilisers, in the production of which the price of gas occupies a significant part, and, as a consequence, cause an increase in the cost of food products. To prevent this, the government is imposing temporary restrictions on the exports," he added.',
                ('He said that the quota for nitrogen fertilisers', 'EI'),
                'would be set at 5.9 million tonnes and complex nitrogen-containing fertilisers at ',
                ('5.35 tonnes.', 'EI')
            )
            st.markdown('<br />', unsafe_allow_html=True)
            annotated_text(
                ('Prime Minister Mikhail Mishustin said on Wednesday that Russia would limit exports of nitrogen fertilisers and complex nitrogen-containing fertilisers for the six months to try to curb any further increase in food prices due to higher gas prices.', 'changed'),
                ('"The increase in natural gas prices has had a negative impact on the global markets," Mishustin told a government meeting.', 'unchanged'),
                ('"This could lead to higher prices and shortages of nitrogen fertilisers, in the production of which the price of gas occupies a significant part, and, as a consequence, cause an increase in the cost of food products.', 'unchanged'),
                ('"To prevent this, the government is imposing temporary restrictions on the exports," he added.', 'unchanged'),
                ('He said that the quota for nitrogen fertilisers would be set at 5.9 million tonnes and complex nitrogen-containing fertilisers at 5.35 tonnes.', 'changed')
            )
            st.markdown('<br />', unsafe_allow_html=True)


if __name__ == '__main__':
    main()