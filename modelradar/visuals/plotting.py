import pandas as pd
import plotnine as p9


class Plots:

    @staticmethod
    def error_distribution_baseline(df: pd.DataFrame,
                                    baseline: str,
                                    thr: float):
        plot = p9.ggplot(df) + \
               p9.aes(x=baseline) + \
               p9.geom_histogram(alpha=.95,
                                 bins=30,
                                 color='black',
                                 fill='#69a765') + \
               Plots.get_theme() + \
               p9.geom_vline(xintercept=thr,
                             colour='red',
                             size=1) + \
               p9.labs(x=f'Error distribution of {baseline}',
                       y='Count')

        return plot

    @classmethod
    def average_error_by_freq(cls, df: pd.DataFrame):
        # avg_error = df.groupby('Model')['Error'].mean()
        # order = avg_error.sort_values(ascending=True).index.tolist()
        df['Model'] = pd.Categorical(df['Model'], categories=cls.ORDER[::-1])

        # plot = \
        #     p9.ggplot(data=df,
        #               mapping=p9.aes(x='Frequency',
        #                              y='Error',
        #                              group='Model',
        #                              fill='Model')) + \
        #     p9.facet_grid('~Frequency') + \
        #     p9.geom_bar(position='dodge',
        #                 stat='identity',  # color='Color',
        #                 width=0.9) + \
        #     Plots.get_theme() + \
        #     p9.labs(x='Sampling frequency', y='Error') + \
        #     p9.scale_fill_manual(values=COLOR_MAP)

        plot = p9.ggplot(data=df,
                         mapping=p9.aes(x='Model',
                                        y='Error',
                                        group='Frequency',
                                        fill='Model')) + \
               p9.facet_grid('~Frequency') + \
               p9.geom_bar(position='dodge',
                           stat='identity',
                           width=0.9) + \
               Plots.get_theme() + \
               p9.theme(axis_text_x=p9.element_text(angle=60, size=7),
                        strip_text=p9.element_text(size=10)) + \
               p9.labs(x='', y='SMAPE') + \
               p9.scale_fill_manual(values=cls.COLOR_MAP) + \
               p9.guides(fill=None)

        return plot



    @classmethod
    def average_error_by_stationarity(cls, df: pd.DataFrame, colname: str):
        df = df.rename(columns={'variable':'Model'})

        df['Model'] = pd.Categorical(df['Model'], categories=cls.ORDER[::-1])

        plot = p9.ggplot(data=df,
                         mapping=p9.aes(x='Model',
                                        y='value',
                                        group=colname,
                                        fill='Model')) + \
               p9.facet_grid(f'~{colname}') + \
               p9.geom_bar(position='dodge',
                           stat='identity',
                           width=0.9) + \
               Plots.get_theme() + \
               p9.theme(axis_text_x=p9.element_text(angle=60, size=7),
                        strip_text=p9.element_text(size=10)) + \
               p9.labs(x='', y='SMAPE') + \
               p9.scale_fill_manual(values=cls.COLOR_MAP) + \
               p9.guides(fill=None)

        return plot


    @classmethod
    def error_dist_by_model(cls, df: pd.DataFrame):
        # avg_error = df.groupby('Model')['Error'].median()
        avg_error = df.groupby('Model')['Error'].mean()
        # order = avg_error.sort_values(ascending=False).index.tolist()

        df_melted = df.sort_values('Error', ascending=False).reset_index(drop=True)
        df_melted['Model'] = pd.Categorical(df_melted['Model'].values.tolist(),
                                            categories=cls.ORDER)

        # plot = p9.ggplot(df_melted,
        #                  p9.aes(x='Model',
        #                         y='Error')) + \
        #        Plots.get_theme() + \
        #        p9.geom_boxplot(fill='#66CDAA',
        #                        width=0.7,
        #                        show_legend=False) + \
        #        p9.coord_flip() + \
        #        p9.labs(x='Error distribution')  # + \
        # p9.geom_hline(data=avg_error.reset_index(),
        #               mapping=p9.aes(yintercept='Error'),
        #               colour='red',
        #               size=1)

        plot = p9.ggplot(df_melted,
                         p9.aes(x='Model',
                                y='Error', fill='Model')) + \
               Plots.get_theme() + \
               p9.geom_boxplot(
                               width=0.7,
                               show_legend=False) + \
               p9.coord_flip() + \
               p9.scale_fill_manual(values=cls.COLOR_MAP)+ \
               p9.labs(x='Error distribution') + \
               p9.guides(fill=None)

        return plot

