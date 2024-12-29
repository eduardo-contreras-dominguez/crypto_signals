import pandas as pd
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

from objects.data_manipulation.covariance import Covariance
from loguru import logger
from tqdm import tqdm

from utils.helpers.pandas_helpers import keep_levels, index_slice
class PCA:
    @staticmethod
    def rolling_pca(prices_df, windows, covariance_mode = 'simple'):
        """
        Perform a rolling PCA on the given prices DataFrame for the specified windows.

        Args:
        - prices_df (pd.DataFrame): DataFrame containing prices data for multiple assets.
        - windows (list): List of rolling window sizes to compute PCA.

        Returns:
        - pd.DataFrame: DataFrame with MultiIndex (symbol, period, field) containing the
                        original price, fair price, and difference between price and fair price.
        """
        logger.info('Starting rolling PCA computation')
        results = []
        df_for_pca = prices_df.dropna(how='any')
        all_dfs = []
        # Iterate through each rolling window size
        for window in windows:
            logger.info(f'Processing rolling PCA for window size {window}')

            # Create a matrix to store reconstructed fair prices for each window
            fair_prices_matrix = []

            # Loop through the price data with the rolling window
            for end_idx in tqdm(range(window, len(df_for_pca)), desc=f"Window {window}", leave=False):
                start_idx = end_idx - window
                rolling_prices = df_for_pca.iloc[start_idx:end_idx]

                # Step 1: Normalize the prices for the current window (using numpy)
                mean = np.mean(rolling_prices, axis=0)
                std = np.std(rolling_prices, axis=0)
                normalized_prices = (rolling_prices - mean) / std

                # Step 2: Compute the covariance matrix of the normalized prices
                if covariance_mode == 'simple':
                    covariance_matrix = Covariance.simple(normalized_prices)
                elif covariance_mode == 'exponential':
                    covariance_matrix = Covariance.exponential(normalized_prices)

                # Step 3: Calculate eigenvalues and eigenvectors of the covariance matrix
                eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

                # Step 4: Sort the eigenvalues and eigenvectors by the eigenvalues in descending order
                sorted_indices = np.argsort(eigenvalues)[::-1]
                eigenvalues_sorted = eigenvalues[sorted_indices]
                eigenvectors_sorted = eigenvectors[:, sorted_indices]

                # Step 5: Select the number of components that explain at least 80% of the variance
                cumulative_variance = np.cumsum(eigenvalues_sorted) / np.sum(eigenvalues_sorted)
                num_components = np.argmax(cumulative_variance >= 0.80) + 1
                if num_components == 0:
                    logger.warning(f"No components retained for window {window} at index {end_idx}")
                    continue

                # Step 6: Project the normalized prices onto the selected eigenvectors
                selected_eigenvectors = eigenvectors_sorted[:, :num_components]
                transformed_data = np.dot(normalized_prices, selected_eigenvectors)

                # Step 7: Reconstruct the prices using the selected eigenvectors and eigenvalues
                reconstructed = np.dot(transformed_data, selected_eigenvectors.T)

                # Step 8: Rescale the reconstructed prices back to the original scale
                fair_prices =reconstructed * std.values + mean.values   # Reverse the normalization

                # Append the reconstructed fair prices for each asset at this window
                fair_prices_matrix.append(fair_prices[-1])  # Only keep the last row of fair prices

            # Prepare results after processing all windows for the current window size
            fair_prices_matrix = np.array(fair_prices_matrix)
            for asset_idx, asset in enumerate(prices_df.columns):
                current_df = pd.DataFrame(index = df_for_pca[asset].iloc[window:].index, columns = ['symbol', 'period', 'price', 'fair_price', 'difference'])
                current_df['period'] = window
                current_df['symbol'] = asset
                current_df['price'] = df_for_pca[asset].iloc[window:].values
                current_df['fair_price'] = fair_prices_matrix[:, asset_idx]
                current_df['difference'] = current_df['price'] - current_df['fair_price']
                current_df_melted = current_df.melt(id_vars=['symbol', 'period'],
                                                    value_vars=['price', 'fair_price', 'difference'],
                                                    var_name='field',
                                                    value_name='value',
                                                    ignore_index=False
                                                    )
                all_dfs.append(current_df_melted)

        # Create the final DataFrame with MultiIndex
        results_df = pd.concat(all_dfs)
        #TODO: be more flexible with the name of the index
        pivot_table = results_df.reset_index().pivot_table(
            index='timestamp',
            columns=['symbol', 'period', 'field'],
            values='value'
        )

        logger.success('Rolling PCA computation completed')
        return pivot_table

def main():
    investment_universe = [
        'XRP/USDC:USDC',
        'SOL/USDC:USDC',
        'ADA/USDC:USDC',
        'BTC/USDC:USDC',
        'ETH/USDC:USDC']
    df = pd.read_parquet('/Users/educontreras/PycharmProjects/crypto_signals/objects/retriever/hourly_prices.parquet')
    df.columns = df.columns.rename("field", level=0)
    close_df = keep_levels(index_slice(df, field='close'), ['symbol'])

    df_subset = close_df.loc[:, investment_universe]
    returns_df = df_subset.pct_change().dropna(how='any')
    pca_signal = PCA.rolling_pca(df_subset, windows=[100, 200, 300])
if __name__ == '__main__':
    main()