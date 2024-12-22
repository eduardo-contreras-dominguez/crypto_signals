from objects.data_manipulation.splitter import Splitter
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from utils.helpers.pandas_helpers import keep_levels,index_slice
from loguru import logger
import numpy as np
import pandas as pd


class MLSignalSelector:
    """
    Class to use machine learning algorithms (e.g., Random Forest) to combine multiple signals and predict asset movements.
    """

    @staticmethod
    def random_forest(signal_df: pd.DataFrame, returns_df: pd.DataFrame, window_level: str, asset_level: str,
                      test_size: float = 0.2) -> tuple:
        """
        Train a Random Forest model individually for each asset and calculate positions for each.

        Args:
        - signal_df (pd.DataFrame): DataFrame with signals for each asset.
        - returns_df (pd.DataFrame): DataFrame with returns for each asset, used as the target for prediction.
        - window_level (str): The level of MultiIndex where the window information is stored.
        - asset_level (str): The level of MultiIndex where the asset information is stored.
        - test_size (float): The proportion of the data to be used for the test set (default is 0.2).

        Returns:
        - pd.DataFrame: DataFrame with recommended positions for each asset.
        - dict: Dictionary containing model evaluation statistics for each asset.
        """

        # Prepare a DataFrame to hold the final positions
        # Ensure the signals and returns are aligned by index (dates)
        logger.info('Combining signals - Random Forest')
        signal_df, returns_df = signal_df.align(returns_df, axis=0, join='inner')

        # Prepare a DataFrame to hold the final positions
        all_positions = pd.DataFrame(index=signal_df.index,
                                     columns=signal_df.columns.get_level_values(asset_level).unique())

        stats = {}
        # Train and evaluate a model for each asset individually
        for asset in signal_df.columns.get_level_values(asset_level).unique():
            logger.info(f'Retrieving for asset {asset}')
            asset_signal = keep_levels(index_slice(signal_df, **{asset_level: asset}), window_level)
            asset_returns = returns_df[asset]

            # Align signals and returns
            features, target = asset_signal.align(asset_returns.shift(-1).dropna(),axis = 0, join="inner")

            # Remove rows with NaNs in features or target
            valid_rows = features.notna().all(axis=1) & target.notna()
            features = features[valid_rows]
            target = target[valid_rows]

            # Convert target to binary classification: 1 if return > 0, -1 otherwise
            target = 2 * (target > 0).astype(int) - 1

            # Use the Splitter to divide the data into train and test sets
            X_train, X_test, y_train, y_test = Splitter.split_data(features, target, test_size)

            # Train the Random Forest model
            rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            rf_model.fit(X_train, y_train)

            # Evaluate the model on the test set
            predictions = rf_model.predict(X_test)
            accuracy = (predictions== y_test).mean()
            logger.info(f'Accuracy for {asset} --> {round(accuracy * 100, 1)}')
            stats[asset] = {"accuracy": accuracy, "predicted_positions": predictions}

            # Make predictions on the entire dataset and calculate positions
            all_predictions = rf_model.predict(features)
            positions = pd.Series(all_predictions, index=features.index)

            # Assign positions based on predicted return
            all_positions[asset] = positions

        return all_positions, stats

