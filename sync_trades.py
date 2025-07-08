#!/usr/bin/env python3
"""
Sync trades with database for web interface
"""

import sys
import json
import os
from datetime import datetime
sys.path.append('src')

try:
    from src.trading.alpaca_tracker import AlpacaTracker
    
    # Initialize tracker
    tracker = AlpacaTracker()
    
    # Sync trades
    tracker.sync_trades()
    
    # Get updated order data
    recent_orders = tracker.get_orders(limit=10)
    
    # Format for web consumption
    result = {
        'success': True,
        'message': 'Trades synced successfully',
        'data': {
            'recent_orders': recent_orders,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    print(json.dumps(result, indent=2, default=str))
    
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e),
        'message': 'Failed to sync trades'
    }
    print(json.dumps(error_result, indent=2))
    sys.exit(1) 