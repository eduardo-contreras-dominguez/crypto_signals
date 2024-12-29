from loguru import logger
import pandas as pd
from utils.helpers.pandas_helpers import index_slice,keep_levels
class Statistics:
    @staticmethod
    def autocorrelation(
        returns_df,
        asset_level='symbol',
        returns_window_level=None,
        windows=[1, 5, 10],
        autocorr_window_level='autocorr_window'
    ):
            """
            Compute the autocorrelation of returns for different rolling windows.

            Args:
            - returns_df (pd.DataFrame): DataFrame with MultiIndex or regular columns representing assets/returns.
            - asset_level (str): Name to use for the asset level in the output.
            - returns_window_level (str or None): Name of the level that represents return windows in the input DataFrame.
                                                  If None, assume columns directly represent assets.
            - windows (list): List of rolling window sizes to compute the autocorrelations.
            - autocorr_window_level (str): Name of the level to use for the autocorrelation windows in the output.

            Returns:
            - pd.DataFrame: DataFrame with MultiIndex containing autocorrelation values for each asset and autocorrelation window.
            """
            logger.info("Starting autocorrelation computation")
            autocorrelations = []

            for window in windows:
                logger.info(f"Computing autocorrelation for window size {window}")

                for asset in returns_df.columns if asset_level is None else returns_df.columns.get_level_values(asset_level).unique():
                    for returns_window in returns_df.columns.get_level_values(returns_window_level).unique():
                        # Select the returns series for the current asset
                        if returns_window_level is None or asset_level is None:
                            returns_series = returns_df[asset]
                        else:
                            if asset_level is not None:
                                returns_series = keep_levels(index_slice(returns_df, **{returns_window_level : returns_window, asset_level : asset}), [asset_level])

                        # Compute the rolling autocorrelation
                        autocorr = returns_series.rolling(window=window).apply(lambda x: x.autocorr(), raw=False).dropna()

                        if autocorr.empty:
                            logger.warning(f'Not enough data for {asset} - window {window}')
                        # Store the results with the appropriate indices
                        autocorrelations.append(pd.DataFrame({
                            'timestamp': autocorr.index,
                            'value': autocorr.values.flatten(),
                            autocorr_window_level: window,
                            asset_level: asset
                        }))

            # Concatenate all autocorrelation results
            autocorr_df = pd.concat(autocorrelations)

            # Reset index and pivot if needed
            if returns_window_level is not None:
                autocorr_df = autocorr_df.reset_index().pivot_table(
                    index='timestamp',
                    columns=[asset_level, autocorr_window_level],
                    values='value'
                )

            logger.success("Autocorrelation computation completed")
            return autocorr_df