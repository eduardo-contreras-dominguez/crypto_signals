import ccxt
import pandas as pd
from config.connection import PRIVATE_KEY, WALLET


def fetch_historical_data(symbol: str, exchange: ccxt.Exchange,start_date = '2014-01-01', timeframe: str = '5m', max_limit: int = 1000):
    """
    Recovers historical data from a given exchange and a given cryptocurrency.

    Args:
        symbol (str): The market symbol (e.g., 'BTC/USDT').
        exchange (ccxt.Exchange): The exchange object to perform the query.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        timeframe (str): The time interval for the data (e.g., '1m', '5m', '1h').
        max_limit (int): The maximum number of candles to retrieve per request.

    Returns:
        pd.DataFrame: DataFrame with the retrieved historical data.
    """

    from_ts = exchange.parse8601(f'{start_date}T00:00:00Z')  # Start date

    ohlcv_list = []

    while True:
        # Realize the request with a limit
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=from_ts, limit=max_limit)
        ohlcv_list.extend(ohlcv)  # Add data recovered to the list
        if len(ohlcv) < max_limit:
            break

        # Update last availables date
        from_ts = ohlcv[-1][0]  # Last timestamp from the candle recovered

    # Data to pandas
    df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convertir timestamp a datetime
    df.sort_values(by='timestamp', inplace=True)  # Ordenar por timestamp

    return df

def main():
    dex = ccxt.hyperliquid({
                'apiKey': PRIVATE_KEY,
                'secret': WALLET,
                'enableRateLimit': True
            })
    symbol = 'PUMP/USDC'
    df = fetch_historical_data(symbol, dex, timeframe='5m')
    print(df.head())

if __name__ == '__main__':
    main()