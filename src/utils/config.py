"""
Loads all API keys from a .env file in the project root.
Make sure you have a `.env` (gitignored) with lines like:

ALPHAVANTAGE_KEY=your_alpha_vantage_key
BINANCE_US_API_KEY=UVmgRMxKoetKkVgEEcuoPhmjGSBgtY3OfhA5Gl9jPFcDpD7LAcs7btnPVJTyqXnf
BINANCE_US_SECRET=5JitR0QMrk8JcATQ1wgvu1jK1fEKwbzt0SDRUXAN4bG2ItvEilb3sFTEzg0aFq0N
NEWSAPI_KEY=your_newsapi_key
ALPACA_API_KEY=PKH6HJ2RBVZ20P8EJPNT
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
FRED_API_KEY=56JBx7QuGHquzDi6yzMd
"""

import os
from dotenv import load_dotenv

# load environment variables from .env
load_dotenv()

ALPHAVANTAGE_KEY   = os.getenv("ALPHAVANTAGE_KEY")
BINANCE_US_API_KEY = os.getenv("BINANCE_US_API_KEY")
BINANCE_US_SECRET  = os.getenv("BINANCE_US_SECRET")
NEWSAPI_KEY        = os.getenv("NEWSAPI_KEY")
ALPACA_API_KEY     = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY  = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL    = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
FRED_API_KEY       = os.getenv("FRED_API_KEY")
