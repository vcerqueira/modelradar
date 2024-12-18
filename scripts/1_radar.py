import pandas as pd
import numpy as np

from utilsforecast.losses import smape, mape

from modelradar.evaluate.radar import ModelRadar
from modelradar.visuals.plotter import ModelRadarPlotter

cv = pd.read_csv('assets/cv.csv')
# cv['is_anomaly'] = cv['is_anomaly'].map({0:'No anomalies',1:'With anomalies'})

radar = ModelRadar(cv_df=cv,
                   freq='ME',
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
error_on_anomalies = radar.evaluate_by_group(group_col='is_anomaly')

error_on_trend = radar.evaluate_by_group(group_col='trend_str')
error_on_seas = radar.evaluate_by_group(group_col='seasonal_str')

# distribution of errors
plot = ModelRadarPlotter.error_distribution(data=err, model_cats=radar.model_order)

plot.save('test.pdf')

# erro geral
# expected shortfall
# erro por horizon bound
# erro em anomalias
# erro em dados dificeis
# high trend strength
# high seas strength

# correr experiências extensas


overall = eval_overall
shortfall = radar.uid_accuracy.expected_shortfall(err)
eval_hbounds
on_anomalies = error_on_anomalies[1]
on_anomalies.name = 'On anomalies'
scores_on_hard = err_hard.mean()
scores_on_hard.name = 'On hard'
error_on_trend
error_on_seas

df = pd.concat([overall,
                shortfall,
                eval_hbounds,
                on_anomalies,
                scores_on_hard,
                error_on_trend,
                error_on_seas], axis=1)

df = df.reset_index().rename(columns={'index': 'Model'})
plot_df = df.melt('Model')


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


plot_df['normalized_value'] = plot_df.groupby('variable')['value'].transform(normalize)

# from plotnine import *
import plotnine as p9


plot = p9.ggplot(data=plot_df,
                 mapping=p9.aes(x='variable',
                                y='normalized_value',
                                group='Model',
                                color='Model')) + \
       p9.geom_line(size=1, alpha=0.8) + \
       p9.geom_point(size=3) + \
       p9.theme_minimal() + \
       p9.labs(title='', y='Normalized Value', x='') + \
       p9.theme(figure_size=(10, 6),
                plot_title=p9.element_text(size=14, face="bold"),
                axis_text_x=p9.element_text(angle=45, hjust=1),
                legend_position="right")
#+ p9.scale_color_brewer(type='qual', palette='Set2'))


plot.save('test.pdf')


#

variables = plot_df['variable'].unique()
models = plot_df['Model'].unique()

# Calculate angles for the radar plot
angles = np.linspace(0, 2 * np.pi, len(variables), endpoint=False)

radar_df = pd.DataFrame()

for model in models:
    model_data = plot_df[plot_df['Model'] == model]
# Add the first point again to close the polygon
values = list(model_data['normalized_value']) + [model_data['normalized_value'].iloc[0]]
angles_plot = list(angles) + [angles[0]]

# Convert polar coordinates to cartesian
x_coords = [v * np.cos(a) for v, a in zip(values, angles_plot)]
y_coords = [v * np.sin(a) for v, a in zip(values, angles_plot)]

temp_df = pd.DataFrame({
    'x': x_coords,
    'y': y_coords,
    'Model': model,
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

plot = (ggplot() +
# Add background circles
geom_path(circle_df, aes(x='x', y='y', group='r'),
          color='grey', alpha=0.3, linetype='dashed') +
# Add axis lines
geom_path(axis_df, aes(x='x', y='y', group='angle'),
          color='grey', alpha=0.3) +
# Add model lines
geom_path(radar_df, aes(x='x', y='y', group='Model', color='Model'),
          size=1, alpha=0.8) +
# Add points
geom_point(radar_df[radar_df['group'] != len(variables)],
           aes(x='x', y='y', color='Model'), size=3) +
# Add variable labels
annotate('text',
         x=[1.1 * np.cos(a) for a in angles],
         y=[1.1 * np.sin(a) for a in angles],
         label=variables,
         size=10) +
# Customize the plot
coord_fixed(ratio=1) +
theme_minimal() +
theme(
    axis_text=element_blank(),
    axis_title=element_blank(),
    panel_grid=element_blank(),
    plot_title=element_text(size=14, face="bold"),
    figure_size=(10, 10)
) +
labs(title='Model Performance Comparison') +
scale_color_brewer(type='qual', palette='Set2'))

plot.save('test.pdf')
