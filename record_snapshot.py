#!/usr/bin/env python3
"""
Record portfolio snapshot for web interface
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
    
    # Record snapshot
    tracker.record_portfolio_snapshot()
    
    # Get updated data
    account_info = tracker.get_account_info()
    
    # Format for web consumption
    result = {
        'success': True,
        'message': 'Portfolio snapshot recorded successfully',
        'data': {
            'account_info': account_info,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    print(json.dumps(result, indent=2, default=str))
    
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e),
        'message': 'Failed to record portfolio snapshot'
    }
    print(json.dumps(error_result, indent=2))
    sys.exit(1) 