"""
Loads all API keys from a .env file in the project root.
Make sure you have a `.env` (gitignored) with lines like:

ALPHAVANTAGE_KEY=your_alpha_vantage_key
BINANCE_US_API_KEY=your_binance_us_api_key
BINANCE_US_SECRET=your_binance_us_secret
NEWSAPI_KEY=your_newsapi_key
"""

import os
from dotenv import load_dotenv

# load environment variables from .env
load_dotenv()

ALPHAVANTAGE_KEY   = os.getenv("ALPHAVANTAGE_KEY")
BINANCE_US_API_KEY = os.getenv("BINANCE_US_API_KEY")
BINANCE_US_SECRET  = os.getenv("BINANCE_US_SECRET")
NEWSAPI_KEY        = os.getenv("NEWSAPI_KEY")
