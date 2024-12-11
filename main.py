import websockets
import json
import asyncio
from loguru import logger
from datetime import datetime

async def handle_price_updates(websocket):
    try:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            timestamp = datetime.now().isoformat()
            logger.info(f'[{timestamp}] new prices received: {data}')

    except websoc
