import numpy as np
import pandas as pd


class Covariance:

    @staticmethod
    def simple(df):
        """
        Calculate the simple covariance matrix for the given DataFrame of prices.

        Args:
        - df (pd.DataFrame): DataFrame containing price data.

        Returns:
        - np.ndarray: Covariance matrix of the price data.
        """
        # Check for NaN values and drop them
        df_cleaned = df.dropna(how='any')

        # Calculate and return the simple covariance matrix
        return np.cov(df_cleaned.T)

    @staticmethod
    def exponential(df, decay_rate=0.1):
        """
        Calculate the covariance matrix with exponential decay weights for the given DataFrame of prices.

        Args:
        - prices_df (pd.DataFrame): DataFrame containing price data.
        - decay_rate (float): Decay rate for the exponential weighting (default is 0.1).

        Returns:
        - np.ndarray: Covariance matrix with exponential decay weighting.
        """
        # Ensure there are no NaN values
        df_cleaned = df.dropna(how='any')

        # Number of observations
        n = len(df_cleaned)

        # Create an array of weights based on exponential decay
        weights = np.exp(-decay_rate * np.arange(n))[::-1]  # Reverse so that recent observations have higher weight

        # Normalize weights
        weights /= np.sum(weights)

        # Calculate the weighted covariance matrix
        weighted_covariance_matrix = np.cov(df_cleaned.T, aweights=weights)

        return weighted_covariance_matrix

