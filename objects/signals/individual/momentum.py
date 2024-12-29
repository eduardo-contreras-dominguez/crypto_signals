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

        all_ = []
        for column in prices_df.columns:
            logger.info(f'ASSET - {column}')
            for window in windows:
                signal_df = pd.DataFrame(columns=['timestamp', 'value', 'symbol', 'signal'])
                signal_df['value'] = prices_df[column].rolling(window=window).mean().values.flatten()
                signal_df['timestamp'] = prices_df[column].index.tolist()
                signal_df['symbol'] = column
                signal_df['signal'] = f'sma_{window}'
                all_.append(signal_df)
        output_df = pd.concat(all_)
        pivot_table = output_df.pivot_table(
            index='timestamp',
            columns=['symbol', 'signal'],
            values='value'
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
        all_ = []
        for column in prices_df.columns:
            logger.info(f'ASSET - {column}')
            for window in windows:
                signal_df = pd.DataFrame(columns=['timestamp', 'value', 'symbol', 'signal'])
                signal_df['value'] = prices_df[column].ewm(span=window, adjust=False).mean().values.flatten()
                signal_df['timestamp'] = prices_df[column].index.tolist()
                signal_df['symbol'] = column
                signal_df['signal'] = f'ema_{window}'
                all_.append(signal_df)
        output_df = pd.concat(all_)
        pivot_table = output_df.pivot_table(
            index='timestamp',
            columns=['symbol', 'signal'],
            values='value'
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
        all_ = []
        for column in prices_df.columns:
            logger.info(f'ASSET - {column}')
            for window in windows:
                signal_df = pd.DataFrame(columns=['timestamp', 'value', 'symbol', 'signal'])
                half_length = int(window / 2)
                sqrt_length = int(np.sqrt(window))
                wma_half = prices_df[column].rolling(window=half_length).mean()
                wma_full = prices_df[column].rolling(window=window).mean()
                signal_df['value'] = (2 * wma_half - wma_full).rolling(window=sqrt_length).mean().values.flatten()
                signal_df['timestamp'] = prices_df[column].index.tolist()
                signal_df['symbol'] = column
                signal_df['signal'] = f'hma_{window}'
                all_.append(signal_df)
        output_df = pd.concat(all_)
        pivot_table = output_df.pivot_table(
            index='timestamp',
            columns=['symbol', 'signal'],
            values='value'
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
        all_ = []
        for column in prices_df.columns:
            logger.info(f'ASSET - {column}')
            for window in windows:
                signal_df = pd.DataFrame(columns=['timestamp', 'value', 'symbol', 'signal'])
                delta = prices_df[column].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                signal_df['value'] = 100 - (100 / (1 + rs)).values.flatten()
                signal_df['symbol'] = column
                signal_df['signal'] = f'rsi_{window}'
                signal_df['timestamp'] = rs.index.tolist()
                all_.append(signal_df)
        output_df = pd.concat(all_)
        pivot_table = output_df.pivot_table(
            index='timestamp',
            columns=['symbol', 'signal'],
            values='value'
        )
        logger.success('Success computing RSI signals')
        return pivot_table
    @staticmethod
    def tstat(returns_df, windows):
        logger.info('Computing momentum tstat signal')
        all_ = []
        for column in returns_df.columns:
            logger.info (f'ASSET - {column}')
            for window in windows:
                logger.info(f'WINDOW - {window}')
                signal_df = pd.DataFrame(columns=['timestamp', 'value', 'symbol', 'signal'])
                rolling_mean = returns_df[column].rolling(window = window).mean()
                rolling_std = returns_df[column].rolling(window = window).std()
                t_stat = (rolling_mean /rolling_std) * np.sqrt(window)
                signal_df['value'] = 2 *norm.cdf(t_stat) - 1
                signal_df['timestamp'] = t_stat.index.tolist()
                signal_df['symbol'] = column
                signal_df['signal'] = f"tstat_{window}"
                all_.append(signal_df)
        output_df = pd.concat(all_)
        pivot_table = output_df.pivot_table(
            index='timestamp',
            columns=['symbol', 'signal'],
            values='value'
        )
        logger.success('Success computing tstat signal')
        return pivot_table