from loguru import logger
import pandas as pd
from utils.helpers.pandas_helpers import keep_levels, index_slice
class SignalSelector:
    @staticmethod
    def select_window(signal_df: pd.DataFrame, window: int, window_level: str, asset_level:str) -> pd.DataFrame:
        """
        Select signals for a specific window (period).

        Args:
        - signal_df (pd.DataFrame): A DataFrame with MultiIndex columns.
        - window (int): The window (period) to select.
        - window_level (str): The name of the MultiIndex level that identifies the window.

        Returns:
        - pd.DataFrame: A DataFrame with signals for the specified window.
                        Columns correspond to assets, and the index represents dates.
        """
        # Use .xs to filter columns at the specific window level
        return keep_levels(signal_df.xs(key=window, level=window_level, axis=1), asset_level)

    @staticmethod
    def average_windows(signal_df: pd.DataFrame, windows: list, window_level: str, asset_level: str) -> pd.DataFrame:
        """
        Calculate the average of signals across multiple windows (periods).

        Args:
        - signal_df (pd.DataFrame): A DataFrame with MultiIndex columns.
        - windows (list): A list of windows (periods) to include in the average.
        - window_level (str): The name of the MultiIndex level that identifies the window.
        - asset_level (str): The name of the MultiIndex level that identifies the asset.

        Returns:
        - pd.DataFrame: A DataFrame with the average of signals across the specified windows.
                        Columns correspond to assets, and the index represents dates.
        """
        # Filter columns for the specified windows
        selected_windows = signal_df.loc[:, signal_df.columns.get_level_values(window_level).isin(windows)]

        # Group columns by the asset level and compute the mean across the selected windows
        return selected_windows.groupby(level=asset_level, axis=1).mean()

    @staticmethod
    def crossover_signal(signal_df: pd.DataFrame, windows: list, window_level: str, asset_level: str) -> pd.DataFrame:
        """
        Calculate a crossover signal between short-term and long-term signals across multiple windows.
        The signal ranges from -1 to 1, where 1 indicates that the shortest window's signal is
        above the longer windows, and -1 indicates the opposite.

        Args:
        - signal_df (pd.DataFrame): A DataFrame with MultiIndex columns containing the signals for each asset.
        - windows (list): A list of windows (periods) to evaluate for the crossover.
        - window_level (str): The name of the MultiIndex level that identifies the window/period.
        - asset_level (str): The name of the MultiIndex level that identifies the asset.

        Returns:
        - pd.DataFrame: A DataFrame with crossover signals between -1 and 1.
                         The columns correspond to assets, and the index represents dates.
        """
        logger.info(f"Computing crossover for individual signals")

        # Initialize an empty list to store the signals
        crossover_signals = pd.DataFrame(index=signal_df.index, columns = signal_df.columns.get_level_values(asset_level).unique())

        # Loop through each asset
        for asset in signal_df.columns.get_level_values(asset_level).unique():
            logger.info(f'ASSET: {asset}')
            # Filter the data for the current asset
            asset_signal = keep_levels(index_slice(signal_df, **{asset_level : asset}), window_level)

            # Initialize the list to store crossover signals for this asset
            asset_crossover_signal = []

            # Iterate through each row (date) of the asset signal
            for _, row in asset_signal.iterrows():
                # Extract values for the current date across all windows
                window_values = row[windows].to_dict()

                # Sort windows based on their period size (ascending order: short-term -> long-term)
                sorted_windows = sorted(window_values.keys())

                # Calculate the crossover signal based on how many short-term windows are above the long-term windows
                signal_value = 0  # Initialize the signal value
                total_windows = len(sorted_windows)
                count_above = 0  # Count how many short-term windows are above the long-term ones

                # Compare short-term windows to long-term windows
                for i in range(total_windows - 1):
                    # Calculate how many windows are above all the others
                    if window_values[sorted_windows[i]] > window_values[sorted_windows[i + 1]]:
                        count_above += 1

                # The signal value ranges between -1 and 1 based on the number of windows above
                signal_value = (count_above / (total_windows - 1)) * 2 - 1  # Normalize between -1 and 1

                # Append the signal value to the list for this asset
                asset_crossover_signal.append(signal_value)

            # Store the calculated crossover signal for the asset
            crossover_signals[asset] = asset_crossover_signal

        return crossover_signals