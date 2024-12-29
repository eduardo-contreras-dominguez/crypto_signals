
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
    @staticmethod
    def log_returns(prices_df, windows, asset_level='symbol', window_level='period'):
        """
        Compute the log returns for a DataFrame of prices over different rolling windows.

        Args:
        - prices_df (pd.DataFrame): DataFrame where each column represents the price of an asset.
        - windows (list): List of rolling window sizes to compute the log returns.
        - asset_level (str): Name to use for the asset level in the output.
        - window_level (str): Name to use for the window level in the output.

        Returns:
        - pd.DataFrame: DataFrame with MultiIndex containing the log returns for each window.
        """
        logger.info('Starting log returns computation')
        log_returns = []

        for window in windows:
            logger.info(f'Computing log returns for window size {window}')

            for asset in prices_df.columns:
                # Compute rolling log returns
                returns = np.log(prices_df[asset]) - np.log(prices_df[asset].shift(window))

                log_returns.append(pd.DataFrame({
                    'value': returns,
                    window_level: window,
                    asset_level: asset
                }))

        log_returns_df = pd.concat(log_returns)

        log_returns_df = log_returns_df.reset_index().pivot_table(
            index='timestamp',
            columns=[asset_level, window_level],
            values='value'
        )
        latest_index = log_returns_df.index[-1]  # Get the last index (most recent timestamp)
        log_returns_df = log_returns_df.dropna(axis=1, subset=[latest_index])
        logger.success('Log returns computation completed')
        return log_returns_df

    @staticmethod
    def pct_returns(prices_df, windows, asset_level='symbol', window_level='period'):
        """
        Compute the percentage returns for a DataFrame of prices over different rolling windows.

        Args:
        - prices_df (pd.DataFrame): DataFrame where each column represents the price of an asset.
        - windows (list): List of rolling window sizes to compute the percentage returns.
        - asset_level (str): Name to use for the asset level in the output.
        - window_level (str): Name to use for the window level in the output.

        Returns:
        - pd.DataFrame: DataFrame with MultiIndex containing the percentage returns for each window.
        """
        logger.info('Starting percentage returns computation')
        percentage_returns = []

        for window in windows:
            logger.info(f'Computing percentage returns for window size {window}')

            for asset in prices_df.columns:
                # Compute rolling percentage returns
                returns = (prices_df[asset] / prices_df[asset].shift(window) - 1) * 100

                percentage_returns.append(pd.DataFrame({
                    'value': returns,
                    window_level: window,
                    asset_level: asset
                }))

        percentage_returns_df = pd.concat(percentage_returns)

        percentage_returns_df = percentage_returns_df.reset_index().pivot_table(
            index='timestamp',
            columns=[asset_level, window_level],
            values='value'
        )

        logger.success('Percentage returns computation completed')
        return percentage_returns_df