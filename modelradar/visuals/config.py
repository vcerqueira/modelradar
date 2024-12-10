import plotnine as p9

THEME = p9.theme_538(base_family='Palatino', base_size=12) + \
        p9.theme(plot_margin=.025,
                 panel_background=p9.element_rect(fill='white'),
                 plot_background=p9.element_rect(fill='white'),
                 legend_box_background=p9.element_rect(fill='white'),
                 strip_background=p9.element_rect(fill='white'),
                 legend_background=p9.element_rect(fill='white'),
                 axis_text_x=p9.element_text(size=9, angle=0),
                 axis_text_y=p9.element_text(size=9),
                 legend_title=p9.element_blank())
