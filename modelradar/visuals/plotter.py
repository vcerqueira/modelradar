from typing import List

import numpy as np
import pandas as pd
import plotnine as p9

from modelradar.visuals.config import THEME
from modelradar.pipelines.utils import LogTransformation


class ModelRadarPlotter:
    MAIN_COL = 'darkgreen'

    @classmethod
    def error_barplot(cls, data: pd.DataFrame, x: str, y: str):
        df = data.sort_values(y, ascending=False).reset_index(drop=True)
        df[x] = pd.Categorical(df[x].values.tolist(), categories=df[x].values.tolist())

        plot = \
            p9.ggplot(data=df,
                      mapping=p9.aes(x=x, y=y)) + \
            p9.geom_bar(position='dodge',
                        stat='identity',
                        width=0.9,
                        fill=cls.MAIN_COL) + \
            THEME + \
            p9.theme(axis_title_y=p9.element_text(size=7),
                     axis_text_x=p9.element_text(size=9)) + \
            p9.labs(x='') + \
            p9.coord_flip() + \
            p9.guides(fill=None)

        return plot

    @staticmethod
    def error_by_horizon_fl(data: pd.DataFrame, model_cats: List[str]):
        data['Model'] = pd.Categorical(data['Model'], categories=model_cats)

        plot = p9.ggplot(data=data,
                         mapping=p9.aes(x='Model',
                                        y='Error',
                                        group='Horizon',
                                        fill='Model')) + \
               p9.facet_grid('~Horizon') + \
               p9.geom_bar(position='dodge',
                           stat='identity',
                           width=0.9) + \
               THEME + \
               p9.theme(axis_text_x=p9.element_text(angle=60, size=7),
                        strip_text=p9.element_text(size=10)) + \
               p9.labs(x='') + \
               p9.guides(fill=None)

        return plot

    @staticmethod
    def error_by_horizon(data: pd.DataFrame, break_interval: int = 3):
        df = data.melt('horizon')
        df = df.rename(columns={'variable': 'Model', 'value': 'Error'})

        max_h = df['horizon'].max()
        breaks = np.unique([1, *np.arange(break_interval, max_h, break_interval), max_h])

        plot = \
            p9.ggplot(df) + \
            p9.aes(x='horizon',
                   y='Error',
                   group='Model',
                   color='Model') + \
            p9.geom_line(size=1) + \
            THEME + \
            p9.scale_x_continuous(breaks=breaks)

        return plot

    @staticmethod
    def winning_ratios(data: pd.DataFrame, reference: str):
        cats = [f'{reference} loses', 'draw', f'{reference} wins']

        data['Result'] = pd.Categorical(data['Result'], categories=cats)

        plot = \
            p9.ggplot(data,
                      p9.aes(fill='Result',
                             y='Probability',
                             x='Model')) + \
            p9.geom_bar(position='stack', stat='identity') + \
            THEME + \
            p9.theme(
                strip_text=p9.element_text(size=12),
                axis_text_x=p9.element_text(size=10, angle=0),
                legend_title=p9.element_blank(),
                legend_position='top') + \
            p9.labs(x='', y='Proportion of probability') + \
            p9.scale_fill_manual(values=['#2E5EAA', '#FCAF38', '#E63946']) + \
            p9.coord_flip()

        return plot

    @staticmethod
    def error_by_group(data: pd.DataFrame, model_cats: List[str]):
        data = data.reset_index()
        data = data.rename(columns={'index': 'Model'})
        data_m = data.melt('Model')

        data_m['Model'] = pd.Categorical(data_m['Model'], categories=model_cats)

        plot = p9.ggplot(data=data_m,
                         mapping=p9.aes(x='Model',
                                        y='value',
                                        group='variable',
                                        fill='Model')) + \
               p9.facet_grid(f'~variable') + \
               p9.geom_bar(position='dodge',
                           stat='identity',
                           width=0.9) + \
               THEME + \
               p9.theme(axis_text_x=p9.element_text(angle=30, size=7),
                        plot_margin=0.025,
                        strip_text=p9.element_text(size=10),
                        strip_background_x=p9.element_text(color='lightgrey'), ) + \
               p9.labs(x='', y='Error') + \
               p9.guides(fill=None)

        return plot

    @staticmethod
    def error_distribution(data: pd.DataFrame, model_cats: List[str], log_transform: bool = False):
        data_melted = data.melt()
        data_melted = data_melted.rename(columns={'variable': 'Model'})

        data_melted['Model'] = pd.Categorical(data_melted['Model'].values.tolist(),
                                              categories=model_cats)

        if log_transform:
            data_melted['value'] = LogTransformation.transform(data_melted['value'])

        plot = p9.ggplot(data_melted,
                         p9.aes(x='Model',
                                y='value', fill='Model')) + \
               THEME + \
               p9.geom_violin(
                   width=0.8,
                   show_legend=False) + \
               p9.labs(y='Error', x='') + \
               p9.coord_flip() + \
               p9.guides(fill=None)

        return plot

    @classmethod
    def multidim_parallel_coords(cls, df: pd.DataFrame, values: str = 'raw'):
        assert values in ['raw', 'normalize', 'rank']

        df = df.reset_index().rename(columns={'index': 'Model'})
        plot_df = df.melt('Model')

        if values == 'normalize':
            y_name = 'Normalized scores'
            plot_df['value'] = plot_df.groupby('variable')['value'].transform(cls._normalize)
        elif values == 'rank':
            y_name = 'Rank scores'
            plot_df['value'] = plot_df.groupby('variable')['value'].rank()
        else:
            y_name = 'Accuracy scores'

        plot_df['variable'] = pd.Categorical(plot_df['variable'], categories=df.columns)

        plot = p9.ggplot(data=plot_df,
                         mapping=p9.aes(x='variable',
                                        y='value',
                                        group='Model',
                                        color='Model')) + \
               p9.geom_line(size=1, alpha=0.8) + \
               p9.geom_point(size=3) + \
               THEME + \
               p9.labs(title='', y=y_name, x='') + \
               p9.theme(figure_size=(10, 6),
                        # plot_title=p9.element_text(size=14, face="bold"),
                        axis_text_x=p9.element_text(angle=45, hjust=1),
                        legend_position="right") + \
               p9.scale_color_brewer(type='qual', palette='Set2')

        return plot

    @staticmethod
    def _normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))


class SpiderPlot:

    @classmethod
    def create_plot(cls, df: pd.DataFrame, models_col: str = 'Model', values: str = 'raw'):
        df = df.reset_index().rename(columns={'index': models_col})
        plot_df = df.melt(models_col)

        if values == 'normalize':
            y_name = 'Normalized scores'
            plot_df['value'] = plot_df.groupby('variable')['value'].transform(cls._normalize)
        elif values == 'rank':
            y_name = 'Rank scores'
            plot_df['value'] = plot_df.groupby('variable')['value'].rank()
            plot_df['value'] /= len(df[models_col].unique())
            plot_df['value'] = 1 - plot_df['value']
        else:
            y_name = 'Accuracy scores'

        plot_df['variable'] = pd.Categorical(plot_df['variable'], categories=df.columns)

        radar_df, circle_df, axis_df, variables, angles = cls._get_plot_data(df=plot_df, models_col=models_col)

        plot = cls._make_plot(radar_df=radar_df,
                              circle_df=circle_df,
                              axis_df=axis_df,
                              variables=variables,
                              angles=angles,
                              y_name=y_name)

        return plot

    @staticmethod
    def _make_plot(radar_df, circle_df, axis_df, variables, angles, y_name):
        plot = p9.ggplot() + \
               p9.geom_path(data=circle_df,
                            mapping=p9.aes(x='x', y='y', group='r'),
                            color='grey',
                            alpha=0.3,
                            linetype='dashed') + \
               p9.geom_path(data=axis_df,
                            mapping=p9.aes(x='x', y='y', group='angle'),
                            color='grey',
                            alpha=0.3) + \
               p9.geom_path(data=radar_df,
                            mapping=p9.aes(x='x', y='y', group='Model', color='Model'),
                            size=1,
                            alpha=0.8) + \
               p9.geom_point(radar_df[radar_df['group'] != len(variables)],
                             p9.aes(x='x', y='y', color='Model'), size=3) + \
               p9.annotate('text',
                           x=[1.1 * np.cos(a) for a in angles],
                           y=[1.1 * np.sin(a) for a in angles],
                           label=variables,
                           size=8) + \
               p9.coord_fixed(ratio=1) + \
               THEME + \
               p9.theme(
                   axis_text=p9.element_blank(),
                   axis_title=p9.element_blank(),
                   plot_margin=0.15,
                   panel_grid=p9.element_blank(),
                   plot_title=p9.element_text(size=14, face="bold"),
                   figure_size=(14, 14)
               ) + \
               p9.scale_color_brewer(type='qual', palette='Set2') + \
               p9.labs(title=y_name)

        return plot

    @classmethod
    def _get_plot_data(cls, df: pd.DataFrame, models_col: str):
        variables = df['variable'].unique()
        models = df[models_col].unique()

        # Calculate angles for the radar plot
        angles = cls._get_angles(len(variables))

        radar_df = pd.DataFrame()
        for model in models:
            model_data = df[df[models_col] == model]

            # Add the first point again to close the polygon
            values = list(model_data['value']) + [model_data['value'].iloc[0]]
            angles_plot = list(angles) + [angles[0]]

            # Convert polar coordinates to cartesian
            x_coords = [v * np.cos(a) for v, a in zip(values, angles_plot)]
            y_coords = [v * np.sin(a) for v, a in zip(values, angles_plot)]

            temp_df = pd.DataFrame({
                'x': x_coords,
                'y': y_coords,
                models_col: model,
                'group': range(len(x_coords))
            })

            radar_df = pd.concat([radar_df, temp_df])

        circle_df = pd.DataFrame()
        for r in np.linspace(0.2, 1, 5):
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_x = r * np.cos(theta)
            circle_y = r * np.sin(theta)
            temp_df = pd.DataFrame({
                'x': circle_x,
                'y': circle_y,
                'r': r
            })
            circle_df = pd.concat([circle_df, temp_df])

        # Create coordinates for the axis lines
        axis_df = pd.DataFrame()
        for angle in angles:
            temp_df = pd.DataFrame({
                'x': [0, np.cos(angle)],
                'y': [0, np.sin(angle)],
                'angle': angle
            })
            axis_df = pd.concat([axis_df, temp_df])

        return radar_df, circle_df, axis_df, variables, angles

    @staticmethod
    def _get_angles(n_variables: int):
        angles = np.linspace(0, 2 * np.pi, n_variables, endpoint=False)

        return angles

    @staticmethod
    def _normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
