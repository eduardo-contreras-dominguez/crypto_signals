
import pandas as pd
import numpy as np
from loguru import logger
class Normalizer:
    @staticmethod
    def compute_z_scores(differences_df, windows, asset_level = 'symbol', window_level = 'period'):
        """
        Compute the Z-Score for the differences over different rolling windows.

        Args:
        - differences_df (pd.DataFrame): DataFrame with MultiIndex (symbol, period) and 'difference' values.
        - windows (list): List of rolling window sizes to compute the Z-Scores.

        Returns:
        - pd.DataFrame: DataFrame with MultiIndex (period, symbol) containing the Z-Scores for each window.
        """
        logger.info('Starting Z-Score computation')
        z_scores = []

        for window in windows:
            logger.info(f'Computing Z-Score for window size {window}')

            # Loop through each symbol (asset)
            for asset in differences_df.columns:
                # Compute rolling mean and std for the difference
                rolling_mean = differences_df[asset].rolling(window=window).mean()
                rolling_std = differences_df[asset].rolling(window=window).std()

                # Compute the Z-Score
                z_score = (differences_df[asset] - rolling_mean) / rolling_std

                # Store the result with the appropriate MultiIndex
                z_scores.append(pd.DataFrame({
                    'value': z_score,
                    window_level: window,
                    asset_level: asset
                }))

        # Concatenate all results into a single DataFrame
        z_scores_df = pd.concat(z_scores)

        # Create MultiIndex (period, symbol)
        z_scores_df = z_scores_df.reset_index().pivot_table(
            index  = 'timestamp',
            columns = [asset_level, window_level],
            values = 'value'
        )

        logger.success('Z-Score computation completed')
        return z_scores_df