import numpy as np
import pandas as pd

class Position:
    @staticmethod
    def mean_reversion(signal_df, long_entry_level, long_exit_level, short_entry_level, short_exit_level, number_days_confirmation=0):
        """
        Generate trading signals based on a mean-reversion strategy with confirmation.

        Args:
        - signal_df (pd.DataFrame): DataFrame containing the signal data.
        - long_entry_level (float): Signal level below which to enter long positions.
        - long_exit_level (float): Signal level above which to exit long positions.
        - short_entry_level (float): Signal level above which to enter short positions.
        - short_exit_level (float): Signal level below which to exit short positions.
        - number_days_confirmation (int): Number of days to confirm the signal before acting. Default is 0.

        Returns:
        - pd.DataFrame: DataFrame with the same shape as `signal_df` containing -1, 0, or 1.
        """
        signals = np.zeros_like(signal_df.values)  # Initialize signals to neutral (0)
        signal_array = signal_df.values
        num_days = signal_array.shape[0]
        num_assets = signal_array.shape[1]

        for day in range(num_days):
            for col in range(num_assets):
                # Check if a position is active
                if day > 0 and signals[day - 1, col] == 1:  # Active long position
                    if signal_array[day, col] > long_exit_level:  # Exit long condition
                        signals[day, col] = 0
                    else:  # Hold the long position
                        signals[day, col] = 1

                elif day > 0 and signals[day - 1, col] == -1:  # Active short position
                    if signal_array[day, col] < short_exit_level:  # Exit short condition
                        signals[day, col] = 0
                    else:  # Hold the short position
                        signals[day, col] = -1

                elif signals[day - 1, col] == 0:  # No active position
                    # Long entry condition
                    if day >= number_days_confirmation and (signal_array[day - number_days_confirmation:day + 1, col] < long_entry_level).all():
                        signals[day, col] = 1  # Enter long if confirmed

                    # Short entry condition
                    elif day >= number_days_confirmation and (signal_array[day - number_days_confirmation:day + 1, col] > short_entry_level).all():
                        signals[day, col] = -1  # Enter short if confirmed

        return pd.DataFrame(signals, index=signal_df.index, columns=signal_df.columns)

    @staticmethod
    def trend_following(
            signal_df,
            long_entry_level,
            long_exit_level,
            short_entry_level,
            short_exit_level,
            weakening_threshold=None,
            number_days_confirmation=0,
            max_holding_period=None,
    ):
        """
        Generate trading signals based on a trend-following strategy with additional exit conditions.

        Args:
        - signal_df (pd.DataFrame): DataFrame containing the signal data.
        - long_entry_level (float): Signal level above which to enter long positions.
        - long_exit_level (float): Signal level below which to exit long positions.
        - short_entry_level (float): Signal level below which to enter short positions.
        - short_exit_level (float): Signal level above which to exit short positions.
        - weakening_threshold (float or None): Percentage drop relative to the previous period to trigger an exit. Default is None.
        - number_days_confirmation (int): Number of days to confirm the signal before acting. Default is 0.
        - max_holding_period (int or None): Max number of periods to hold a position. Default is None (no limit).

        Returns:
        - pd.DataFrame: DataFrame with the same shape as `signal_df` containing -1 (short), 0 (neutral), or 1 (long).
        """
        signals = np.zeros_like(signal_df.values)  # Initialize all signals to neutral (0)
        signal_array = signal_df.values
        num_days = signal_array.shape[0]
        num_assets = signal_array.shape[1]

        for col in range(num_assets):
            entry_day = None  # Track the day when a position is entered

            for day in range(1, num_days):  # Start from day 1 to access previous day
                # Neutral state
                if signals[day - 1, col] == 0:
                    # Long entry
                    if signal_array[day, col] > long_entry_level:
                        # Check confirmation days if applicable
                        if day >= number_days_confirmation and all(
                                signal_array[day - number_days_confirmation:day + 1, col] > long_entry_level
                        ):
                            signals[day, col] = 1  # Enter long
                            entry_day = day  # Mark entry day

                    # Short entry
                    elif signal_array[day, col] < short_entry_level:
                        if day >= number_days_confirmation and all(
                                signal_array[day - number_days_confirmation:day + 1, col] < short_entry_level
                        ):
                            signals[day, col] = -1  # Enter short
                            entry_day = day

                # Long position
                elif signals[day - 1, col] == 1:
                    holding_period = day - entry_day if entry_day is not None else 0

                    # Exit conditions for long
                    if (
                            signal_array[day, col] < long_exit_level or  # Exit if below long exit level
                            (weakening_threshold and
                             (signal_array[day - 1, col] - signal_array[day, col]) / signal_array[
                                 day - 1, col] > weakening_threshold) or
                            (max_holding_period and holding_period >= max_holding_period)
                    # Exit after max holding period
                    ):
                        signals[day, col] = 0  # Exit position
                    else:
                        signals[day, col] = 1  # Continue holding

                # Short position
                elif signals[day - 1, col] == -1:
                    holding_period = day - entry_day if entry_day is not None else 0

                    # Exit conditions for short
                    if (
                            signal_array[day, col] > short_exit_level or  # Exit if above short exit level
                            (weakening_threshold and
                             (signal_array[day, col] - signal_array[day - 1, col]) / abs(
                                        signal_array[day - 1, col]) > weakening_threshold) or
                            (max_holding_period and holding_period >= max_holding_period)
                    # Exit after max holding period
                    ):
                        signals[day, col] = 0  # Exit position
                    else:
                        signals[day, col] = -1  # Continue holding

        return pd.DataFrame(signals, index=signal_df.index, columns=signal_df.columns)

