from typing import Optional

import numpy as np
import pandas as pd

from modelradar.visuals.plotter import ModelRadarPlotter


class RopeAnalysis:
    """Region of Practical Equivalence (ROPE) analysis for model comparison.
    
    This class implements methods to compare model performance using ROPE analysis,
    a Bayesian-inspired approach that allows determining if models are practically
    equivalent or if there is a significant difference in their performance.
    
    The analysis calculates the percentage difference between each model and a reference
    model, then categorizes these differences into three outcomes based on the ROPE threshold:
    - Reference model loses (other model is better)
    - Draw (models are practically equivalent)
    - Reference model wins (reference model is better)
    
    Parameters
    ----------
    rope : float
        The Region of Practical Equivalence threshold as a percentage.
        Differences within ±rope% are considered practically equivalent (draws).
    
    reference : str
        Name of the reference model to compare against. Must be a column
        in the scores DataFrame provided to get_winning_ratios.
        
    Attributes
    ----------
    sides : list
        Formatted strings for the three possible outcomes, with the
        reference model name substituted.
    """

    SIDES = ['{reference} loses', 'draw', '{reference} wins']

    def __init__(self, rope: float, reference: str):
        """Initialize ROPE analysis with threshold and reference model.
        
        Parameters
        ----------
        rope : float
            The ROPE threshold as a percentage.
        
        reference : str
            Name of the reference model.
        """

        self.rope = rope
        self.reference = reference
        self.sides = [side.format(reference=reference) for side in self.SIDES]

    def get_winning_ratios(self,
                           uid_scores: pd.DataFrame,
                           return_plot: bool = False,
                           reference: Optional[str] = None,
                           **kwargs):
        """Calculate winning probability ratios for each model against the reference.
        
        For each model, computes the percentage of cases where it:
        1. Outperforms the reference model by more than the ROPE threshold
        2. Performs equivalently to the reference model (within ROPE)
        3. Underperforms the reference model by more than the ROPE threshold
        
        Parameters
        ----------
        uid_scores : pd.DataFrame
            DataFrame containing error metrics for each unique ID and model.
            Index should be unique IDs and columns should be model names.
            Must include the reference model as a column.
        
        return_plot : bool, default=False
            If True, returns a stacked bar plot visualizing the winning ratios.
            If False, returns a DataFrame with the probability values.
        
        reference : Optional[str], default=None
            Reference model name for the plot title. Required if return_plot is True.
            
        **kwargs
            Additional keyword arguments passed to ModelRadarPlotter.winning_ratios
            when return_plot is True.
        
        Returns
        -------
        pd.DataFrame or plotnine.ggplot
            DataFrame with winning, drawing, and losing probabilities for each model,
            or a plot visualizing these probabilities if return_plot is True.
            
        Notes
        -----
        The reference model itself is excluded from the results.
        """

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
        """Calculate probabilities for the three possible outcomes.
        
        Parameters
        ----------
        diff_vec : pd.Series
            Series of percentage differences between a model and the reference.
        
        Returns
        -------
        tuple
            Three-element tuple containing:
            - Probability that reference model loses (diff < -rope)
            - Probability of a draw (-rope ≤ diff ≤ rope)
            - Probability that reference model wins (diff > rope)
        """

        left = (diff_vec < -self.rope).mean()
        right = (diff_vec > self.rope).mean()
        mid = np.mean([-self.rope < x_ < self.rope for x_ in diff_vec])

        return left, mid, right

    def _calc_percentage_diff(self, scores: pd.DataFrame):
        """Calculate percentage differences between each model and the reference.
        
        Parameters
        ----------
        scores : pd.DataFrame
            DataFrame containing error metrics for each unique ID and model.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with percentage differences between each model and the reference.
            The reference model itself is excluded.
        """

        scores_pd = {}
        for mod in scores.columns:
            if mod == self.reference:
                continue

            scores_pd[mod] = self._percentage_diff(scores[mod], scores[self.reference])

        scores_pd_df = pd.DataFrame(scores_pd, index=scores.index)

        return scores_pd_df

    def _assert_params(self, scores: pd.DataFrame):
        """Verify that the reference model exists in the provided scores.
        
        Parameters
        ----------
        scores : pd.DataFrame
            DataFrame containing model scores.
            
        Raises
        ------
        AssertionError
            If the reference model is not found in scores columns.
        """

        assert self.reference in scores.columns, \
            f'{self.reference} not in scores columns'

    @staticmethod
    def _percentage_diff(x, y):
        """Calculate percentage difference between two values.
        
        Calculates (x - y) / |y| * 100, showing how much x differs from y in percentage terms.
        
        Parameters
        ----------
        x : float or array-like
            The comparison value(s).
        y : float or array-like
            The reference value(s).
        
        Returns
        -------
        float or array-like
            Percentage difference(s) between x and y.
        """
        return ((x - y) / abs(y)) * 100
