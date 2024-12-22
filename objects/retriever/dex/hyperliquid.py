import ccxt  # For interacting with HyperLiquid API
import pandas as pd  # For handling data manipulation and storage in DataFrames
import json  # For handling JSON data serialization/deserialization
from datetime import datetime  # For handling timestamps
from typing import Dict, List  # For type hinting
from loguru import logger
from tqdm import tqdm

from objects.retriever.cacher.s3 import S3
from config.connection import * # Import the configuration file
from utils.retriever import fetch_historical_data
class HyperLiquid:
    def __init__(self):
        """
        Initialize the HyperLiquidCCXT class and connect to the HyperLiquid API using CCXT.
        """
        self.exchange = None
        self.metadata_df = None  # DataFrame to store metadata of available markets
        self._initialize_connection()

    def _initialize_connection(self):
        """
        Load API keys and wallet from the configuration file and initialize the CCXT connection.
        """
        try:
            # data connection
            api_key = PRIVATE_KEY
            secret_key = WALLET

            self.exchange = ccxt.hyperliquid({
                'apiKey': api_key,
                'secret': secret_key,
                'enableRateLimit': True
            })
            logger.info("Successfully initialized connection to HyperLiquid.")
        except Exception as e:
            logger.error(f"Error initializing connection: {e}")
            raise

    def get_available_markets(self):
        """
        Fetch and store all tradeable markets in self.metadata_df.
        The DataFrame will have coins as columns and relevant metadata as indices.
        """
        try:
            markets = self.exchange.fetch_markets()
            market_data = {}

            for market in markets:
                # Only consider active markets
                if market['active']:
                    base_currency = market['base']
                    max_leverage = market['limits'].get('leverage', {}).get('max', None)

                    market_data[base_currency] = {
                        'symbol' : market['symbol'],
                        'quote': market['quote'],
                        'active': market['active'],
                        'max_leverage': max_leverage,
                        'precision': market['precision']
                    }

            # Create the DataFrame with 'base' as columns
            self.metadata_df = pd.DataFrame.from_dict(market_data, orient='index')
            self.metadata_df.sort_index(axis=1)
            logger.info("Successfully fetched and stored market metadata.")
        except Exception as e:
            logger.error(f"Error fetching available markets: {e}")

    def get_historical_prices(self, interval: str = '5m', cache = True, file_name = 'hourly_prices') -> pd.DataFrame:
        """
        Fetch historical price data for all coins available in self.metadata_df at the specified interval.

        Args:
            interval (str): The time interval for the candle data (e.g., "1m", "5m", "1h").
            limit (int): Maximum number of candles to retrieve per request.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where each key is a coin symbol, and the value is a DataFrame
            containing the historical price data for that coin. Each DataFrame includes:
                - timestamp: The timestamp of the candle in datetime format.
                - open: The opening price of the candle.
                - high: The highest price during the candle.
                - low: The lowest price during the candle.
                - close: The closing price of the candle.
                - volume: The traded volume during the candle.
        """
        if self.metadata_df is None or self.metadata_df.empty:
            logger.warning("No market metadata available. Please fetch metadata first.")
            return {}

        historical_prices = []

        for symbol in tqdm(self.metadata_df.symbol.unique()):
            try:
                logger.info(f"Fetching historical prices for {symbol}...")
                current_df = fetch_historical_data(symbol = f"{symbol.split('/')[0]}/USDT", exchange=ccxt.binance(), timeframe = interval, max_limit=1000)
                current_df['symbol'] = symbol
                historical_prices.append(current_df)
                logger.info(f"Successfully fetched historical prices for {symbol}.")
            except Exception as e:
                logger.error(f"Error fetching historical prices for {symbol}: {e}")
        output_df = pd.concat(historical_prices)
        output_df = output_df.pivot_table(index = 'timestamp', columns = 'symbol', values = ['open', 'close', 'high','low','volume'])
        if cache:
            output_df.to_parquet(f'{file_name}.parquet')
            S3.upload_parquet_to_s3(output_df, bucket_name='cryptoprices', object_key = f'{file_name}.parquet')


# Example usage
def main():
    tracker = HyperLiquid()

    # Fetch available markets
    tracker.get_available_markets()
    print(tracker.metadata_df)

    # Fetch historical prices for all available markets
    historical_prices = tracker.get_historical_prices(interval='1h')
    for symbol, df in historical_prices.items():
        print(f"Historical prices for {symbol}:")
        print(df.head())

if __name__ == '__main__':
    main()
