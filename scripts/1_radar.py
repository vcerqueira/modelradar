import pandas as pd

from utilsforecast.losses import smape, mape

from modelradar.evaluate.radar import ModelRadar

cv = pd.read_csv('assets/cv.csv')

radar = ModelRadar(cv_df=cv,
                   freq='ME',
                   metrics=[smape, mape],
                   model_names=None,
                   hardness_reference='SeasonalNaive',
                   ratios_reference='NHITS',
                   rope=10)

err = radar.evaluate(keep_uids=True)
err_hard = radar.uid_accuracy.get_hard_uids(err)
# cv_hard = cv.query(f'unique_id == {radar.uid_accuracy.hard_uid}')

win_ratios_df = radar.rope.get_winning_ratios(err, return_long=True)
radar.rope.get_winning_ratios(err)

radar.rope.get_winning_ratios(err_hard)

radar.uid_accuracy.expected_shortfall(err)
radar.uid_accuracy.expected_shortfall(err, return_df=True)
radar.uid_accuracy.expected_shortfall(err_hard)

eval_overall = radar.evaluate()
# eval_overall = radar.evaluate(return_df=True)
# radar.evaluate(cv=cv_hard.reset_index())
eval_hbounds = radar.evaluate_by_horizon_bounds(return_long=True)

eval_fhorizon = radar.evaluate_by_horizon()
eval_fhorizon.melt('horizon')

radar.evaluate_by_anomaly(anomaly_col='is_anomaly', mode='observations')
radar.evaluate_by_anomaly(anomaly_col='is_anomaly', mode='series')
radar.evaluate_by_group(group_col='is_anomaly')

# error by condition
# distribution of errors

#
from typing import List
import numpy as np
import plotnine as p9
from modelradar.visuals.config import THEME

plot = ModelRadarPlotter.error_barplot(data=eval_overall, x='Model', y='Performance')
plot = ModelRadarPlotter.error_by_horizon_fl(data=eval_hbounds, model_cats=radar.model_order)
plot = ModelRadarPlotter.error_by_horizon(data=eval_fhorizon)
plot = ModelRadarPlotter.winning_ratios(data=win_ratios_df, reference=radar.rope.reference)

plot.save('test.pdf')


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


plot.save('test.pdf')
