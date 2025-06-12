import os
from dotenv import load_dotenv

load_dotenv()  # loads .env in project root

ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY")
BINANCE_API_KEY   = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET    = os.getenv("BINANCE_SECRET")
NEWSAPI_KEY       = os.getenv("NEWSAPI_KEY")
