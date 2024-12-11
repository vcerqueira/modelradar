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
