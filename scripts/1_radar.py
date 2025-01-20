import pandas as pd
import plotnine as p9

from utilsforecast.losses import smape, mape

from modelradar.evaluate.radar import ModelRadar
from modelradar.visuals.plotter import ModelRadarPlotter, SpiderPlot

cv = pd.read_csv('assets/cv.csv')
cv['anomaly_status'] = cv['is_anomaly'].map({0: 'Non-anomalies', 1: 'Anomalies'})

radar = ModelRadar(cv_df=cv,
                   metrics=[smape, mape],
                   model_names=['NHITS', 'MLP', 'MLP1', 'KAN', 'SeasonalNaive'],
                   hardness_reference='SeasonalNaive',
                   ratios_reference='NHITS',
                   rope=10)

err = radar.evaluate(keep_uids=True)
err_hard = radar.uid_accuracy.get_hard_uids(err)
# cv_hard = cv.query(f'unique_id == {radar.uid_accuracy.hard_uid}')

radar.rope.get_winning_ratios(err, return_plot=True, reference=radar.rope.reference)
radar.rope.get_winning_ratios(err)

radar.uid_accuracy.expected_shortfall(err)
radar.uid_accuracy.expected_shortfall(err, return_plot=True)

eval_overall = radar.evaluate()
# eval_overall = radar.evaluate(return_df=True)
# radar.evaluate(cv=cv_hard.reset_index())
eval_hbounds = radar.evaluate_by_horizon_bounds()
plot = radar.evaluate_by_horizon_bounds(return_plot=True, plot_model_cats=radar.model_order)

eval_fhorizon = radar.evaluate_by_horizon()
plot = radar.evaluate_by_horizon(return_plot=True)

radar.evaluate_by_anomaly(anomaly_col='is_anomaly', mode='observations')
radar.evaluate_by_anomaly(anomaly_col='is_anomaly', mode='series')

error_on_anomalies = radar.evaluate_by_group(group_col='anomaly_status')
error_on_trend = radar.evaluate_by_group(group_col='trend_str')
error_on_seas = radar.evaluate_by_group(group_col='seas_str')

# distribution of errors
plot = ModelRadarPlotter.error_distribution(data=err, model_cats=radar.model_order)

df = pd.concat([eval_overall,
                radar.uid_accuracy.expected_shortfall(err),
                eval_hbounds,
                radar.uid_accuracy.accuracy_on_hard(err),
                error_on_anomalies,
                error_on_trend,
                error_on_seas], axis=1)

plot = ModelRadarPlotter.multidim_parallel_coords(df, values='normalize')
# plot.save('assets/examples/test.pdf')

plot = SpiderPlot.create_plot(df=df,
                              values='rank',
                              include_title=False,
                              color_set=None)
plot = plot + p9.theme(plot_margin=0.05,
                       legend_position='top',
                       legend_text=p9.element_text(size=17),
                       legend_key_size=20,
                       legend_key_width=20)
plot.save('assets/examples/test.pdf', width=12, height=16)
