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

radar.rope.get_winning_ratios(err, return_long=True)
radar.rope.get_winning_ratios(err)

radar.rope.get_winning_ratios(err_hard)

radar.uid_accuracy.expected_shortfall(err)
radar.uid_accuracy.expected_shortfall(err, return_df=True)
radar.uid_accuracy.expected_shortfall(err_hard)

eval_overall = radar.evaluate(return_df=True)
# radar.evaluate(cv=cv_hard.reset_index())
eval_hbounds = radar.evaluate_by_horizon_bounds()

eval_fhorizon = radar.evaluate_by_horizon()
eval_fhorizon.melt('horizon')

radar.evaluate_by_anomaly(anomaly_col='is_anomaly', mode='observations')
radar.evaluate_by_anomaly(anomaly_col='is_anomaly', mode='series')
radar.evaluate_by_group(group_col='is_anomaly')

# todo add if condition to turn results into long format for plotting

#

import plotnine as p9
from modelradar.visuals.config import THEME

plot = ModelRadarPlotter.error_barplot(data=eval_overall, x='Model', y='Performance')

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
