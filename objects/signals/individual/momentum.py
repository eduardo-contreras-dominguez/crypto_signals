import numpy as np
import pandas as pd
from scipy.stats import norm
from loguru import logger
class Momentum:
    @staticmethod
    def sma(prices_df, windows):
        """
        Compute the Simple Moving Average (SMA) for each column in prices_df over given windows.

        Args:
        - prices_df (pd.DataFrame): DataFrame containing prices data for multiple assets.
        - windows (list): List of windows for which to calculate the SMA.

        Returns:
        - pd.DataFrame: MultiIndex DataFrame with SMA values for each asset and window.
        """
        logger.info('Computing SMA signals')
        output = []
        for column in prices_df.columns:
            logger.info(f'ASSET - {column}')
            sma_df = pd.DataFrame({window: prices_df[column].rolling(window=window).mean() for window in windows})
            melted = sma_df.melt(var_name='period', value_name='sma', ignore_index=False)
            melted['symbol'] = column
            melted['signal'] = 'sma'
            output.append(melted)
        combined_output = pd.concat(output)
        pivot_table = combined_output.reset_index().pivot_table(
            index=combined_output.index.name,
            columns=['symbol', 'signal', 'period'],
            values='sma'
        )
        logger.success('Success computing SMA signals')
        return pivot_table

    @staticmethod
    def ema(prices_df, windows):
        """
        Compute the Exponential Moving Average (EMA) for each column in prices_df over given windows.

        Args:
        - prices_df (pd.DataFrame): DataFrame containing prices data for multiple assets.
        - windows (list): List of windows for which to calculate the EMA.

        Returns:
        - pd.DataFrame: MultiIndex DataFrame with EMA values for each asset and window.
        """
        logger.info('Computing EMA signals')
        output = []
        for column in prices_df.columns:
            logger.info(f'ASSET - {column}')
            ema_df = pd.DataFrame({window: prices_df[column].ewm(span=window, adjust=False).mean() for window in windows})
            melted = ema_df.melt(var_name='period', value_name='ema', ignore_index=False)
            melted['symbol'] = column
            melted['signal'] = 'ema'
            output.append(melted)
        combined_output = pd.concat(output)
        pivot_table = combined_output.reset_index().pivot_table(
            index=combined_output.index.name,
            columns=['symbol', 'signal', 'period'],
            values='ema'
        )
        logger.success('Success computing EMA signals')
        return pivot_table

    @staticmethod
    def hma(prices_df, windows):
        """
        Compute the Hull Moving Average (HMA) for each column in prices_df over given windows.

        Args:
        - prices_df (pd.DataFrame): DataFrame containing prices data for multiple assets.
        - windows (list): List of windows for which to calculate the HMA.

        Returns:
        - pd.DataFrame: MultiIndex DataFrame with HMA values for each asset and window.
        """
        logger.info('Computing HMA signals')
        output = []
        for column in prices_df.columns:
            logger.info(f'ASSET - {column}')
            hma_df = pd.DataFrame()
            for window in windows:
                half_length = int(window / 2)
                sqrt_length = int(np.sqrt(window))
                wma_half = prices_df[column].rolling(window=half_length).mean()
                wma_full = prices_df[column].rolling(window=window).mean()
                hma_df[window] = (2 * wma_half - wma_full).rolling(window=sqrt_length).mean()
            melted = hma_df.melt(var_name='period', value_name='hma', ignore_index=False)
            melted['symbol'] = column
            melted['signal'] = 'hma'
            output.append(melted)
        combined_output = pd.concat(output)
        pivot_table = combined_output.reset_index().pivot_table(
            index=combined_output.index.name,
            columns=['symbol', 'signal', 'period'],
            values='hma'
        )
        logger.success('Success computing HMA signals')
        return pivot_table

    @staticmethod
    def rsi(prices_df, windows):
        """
        Compute the Relative Strength Index (RSI) for each column in prices_df over given windows.

        Args:
        - prices_df (pd.DataFrame): DataFrame containing prices data for multiple assets.
        - windows (list): List of windows for which to calculate the RSI.

        Returns:
        - pd.DataFrame: MultiIndex DataFrame with RSI values for each asset and window.
        """
        logger.info('Computing RSI signals')
        output = []
        for column in prices_df.columns:
            logger.info(f'ASSET - {column}')
            rsi_df = pd.DataFrame()
            for window in windows:
                delta = prices_df[column].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                rsi_df[window] = 100 - (100 / (1 + rs))
            melted = rsi_df.melt(var_name='period', value_name='rsi', ignore_index=False)
            melted['symbol'] = column
            melted['signal'] = 'rsi'
            output.append(melted)
        combined_output = pd.concat(output)
        pivot_table = combined_output.reset_index().pivot_table(
            index=combined_output.index.name,
            columns=['symbol', 'signal', 'period'],
            values='rsi'
        )
        logger.success('Success computing RSI signals')
        return pivot_table
    @staticmethod
    def tstat(returns_df, windows):
        logger.info('Computing momentum tstat signal')
        output = []
        for column in returns_df.columns:
            logger.info (f'ASSET - {column}')
            ttest_df = pd.DataFrame(columns = windows, index = returns_df.index)
            for window in windows:
                logger.info(f'WINDOW - {window}')
                rolling_mean = returns_df[column].rolling(window = window).mean()
                rolling_std = returns_df[column].rolling(window = window).std()
                t_stat = (rolling_mean /rolling_std) * np.sqrt(window)
                ttest_df[window] = 2 *norm.cdf(t_stat) - 1
                melted = ttest_df.melt(var_name='period', value_name='t_stat', ignore_index=False)
            melted['symbol'] = column
            melted['signal'] = 't_stat'
            output.append(melted)
        combined_output = pd.concat(output)
        pivot_table = combined_output.reset_index().pivot_table(
            index=combined_output.index.name,
            columns=['symbol', 'signal', 'period'],
            values='t_stat'
        )
        logger.success('Success computing tstat signal')
        return pivot_table