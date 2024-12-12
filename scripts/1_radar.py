import pandas as pd

from utilsforecast.losses import smape, mape

import modelradar.evaluate.rope
from modelradar.evaluate.radar import ModelRadar
from modelradar.visuals.plotter import ModelRadarPlotter

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

radar.rope.get_winning_ratios(err, return_plot=True, reference=radar.rope.reference)
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
error_on_group = radar.evaluate_by_group(group_col='is_anomaly')

# distribution of errors

plot = ModelRadarPlotter.error_barplot(data=eval_overall, x='Model', y='Performance')
plot = ModelRadarPlotter.error_by_horizon_fl(data=eval_hbounds, model_cats=radar.model_order)
plot = ModelRadarPlotter.error_by_horizon(data=eval_fhorizon)
plot = ModelRadarPlotter.winning_ratios(data=win_ratios_df, reference=radar.rope.reference)
plot = ModelRadarPlotter.error_by_group(data=error_on_group, model_cats=radar.model_order)
plot = ModelRadarPlotter.error_distribution(data=err, model_cats=radar.model_order)

plot.save('test.pdf')

