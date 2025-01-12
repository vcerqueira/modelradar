from typing import Optional

import numpy as np
import pandas as pd

from modelradar.visuals.plotter import ModelRadarPlotter


class RopeAnalysis:
    SIDES = ['{reference} loses', 'draw', '{reference} wins']

    def __init__(self, rope: float, reference: str):

        self.rope = rope
        self.reference = reference
        self.sides = [side.format(reference=reference) for side in self.SIDES]

    def get_winning_ratios(self,
                           uid_scores: pd.DataFrame,
                           return_plot: bool = False,
                           reference: Optional[str] = None,
                           **kwargs):

        scores_pd = self._calc_percentage_diff(uid_scores)

        # prob_df = scores_pd.apply(lambda x: self._calc_vector_side_probs(x), axis=0).T
        prob_df = scores_pd.apply(self._calc_vector_side_probs, axis=0).T
        prob_df.columns = self.sides

        if return_plot:
            assert reference is not None

            prob_df = prob_df.reset_index()
            prob_df_m = prob_df.melt('index')
            prob_df_m.columns = ['Model', 'Result', 'Probability']

            plot = ModelRadarPlotter.winning_ratios(data=prob_df_m, reference=reference, **kwargs)

            return plot

        return prob_df

    def _calc_vector_side_probs(self, diff_vec: pd.Series):
        left = (diff_vec < -self.rope).mean()
        right = (diff_vec > self.rope).mean()
        mid = np.mean([-self.rope < x_ < self.rope for x_ in diff_vec])

        return left, mid, right

    def _calc_percentage_diff(self, scores: pd.DataFrame):
        scores_pd = {}
        for mod in scores.columns:
            if mod == self.reference:
                continue

            scores_pd[mod] = self._percentage_diff(scores[mod], scores[self.reference])

        scores_pd_df = pd.DataFrame(scores_pd, index=scores.index)

        return scores_pd_df

    def _assert_params(self, scores: pd.DataFrame):
        assert self.reference in scores.columns, \
            f'{self.reference} not in scores columns'

    @staticmethod
    def _percentage_diff(x, y):
        return ((x - y) / abs(y)) * 100
