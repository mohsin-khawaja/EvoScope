#!/usr/bin/env python3
"""
Get Alpaca performance metrics for web interface
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
    
    # Get performance metrics
    metrics = tracker.calculate_performance_metrics()
    
    # Get account info
    account_info = tracker.get_account_info()
    
    # Format for web consumption
    result = {
        'success': True,
        'data': {
            'metrics': metrics,
            'account_info': account_info,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    print(json.dumps(result, indent=2, default=str))
    
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e),
        'data': None
    }
    print(json.dumps(error_result, indent=2))
    sys.exit(1) 