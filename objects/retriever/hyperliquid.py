import pandas as pd
import requests
import websockets
import json
import asyncio
from datetime import datetime
from typing import Dict, List
from loguru import logger


class HyperLiquidPriceTracker:
    def __init__(self):
        self.base_url = 'https://api.hyperliquid.xyz'
        self.meta_url = f'{self.base_url}/info'
        self.candles_url = f'{self.base_url}/candles'
        self.ws_url = 'wss://api.hyperliquid.xyz/ws'
        self.prices: Dict[str, float] = {}

    def get_available_markets(self) -> List[dict]:
        try:
            headers = {
                'Content-Type' : 'application/json',

            }
            payload = {"type" : "meta"}
            response = requests.post(self.meta_url, headers = headers, json = payload)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data.get('universe', []))
            if not df.empty:
                market_dict = {}
                for i, row in df.iterrows():
                    coin_name = row['name']
                    row_dict = row.to_dict()
                    del row_dict['name']
                    market_dict[coin_name] = row_dict
            output_df = pd.DataFrame(market_dict).sort_index(axis = 1)
            return output_df
        except Exception as e:
            logger.error(f'Error requesting for {e}')

    def get_historical_prices(self, coin):
        pass
    async def stream_prices(self, symbols: List[str] = None):
        if symbols is None:
            symbols = ['BTC', 'ETH', 'SOL', 'XRP']
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    subscribe_msg = {
                        'method': 'subscribe',
                        'subscription': {'type': 'l2book', 'coins': symbols}
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    while True:
                        response = await ws.recv()
                        data = json.loads(response)
                        self._process_price_update(data)
            except websockets.exceptions.ConnectionClosed:
                logger.info('Connection closed, reconnecting ...')
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f'Error: {e}')
                await asyncio.sleep(5)

    def _process_price_update(self, data: dict):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            if 'data' in data:
                for update in data['data']:
                    symbol = update.get('coin')
                    if 'asks' in update and update['asks']:
                        ask_price = float(update['asks'][0][0])
                        bid_price = float(update['bids'][0][0])
                        mid_price = (ask_price + bid_price) / 2
                        self.prices[symbol] = mid_price
                        logger.info(f'[{timestamp} {symbol} {mid_price: .2f}')
        except Exception as e:
            logger.error(f'Error processing price data: {e}')


async def main():
    tracker = HyperLiquidPriceTracker()
    logger.info('Current Prices')
    spot_prices = tracker.get_available_markets()
    for asset in spot_prices:
        logger.success(f"Success for {asset['name']}: ${asset['lastPrice']}")


if __name__ == '__main__':
    asyncio.run(main())