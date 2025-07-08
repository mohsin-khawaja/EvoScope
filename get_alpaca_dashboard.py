#!/usr/bin/env python3
"""
Get Alpaca dashboard data for web interface
"""

import sys
import json
import os
sys.path.append('src')

try:
    from src.trading.alpaca_tracker import get_alpaca_dashboard
    
    # Get dashboard data
    dashboard_data = get_alpaca_dashboard()
    
    # Format for web consumption
    result = {
        'success': True,
        'data': {
            'account_info': dashboard_data.get('account_info', {}),
            'positions': dashboard_data.get('positions', []),
            'performance': dashboard_data.get('performance', {}),
            'recent_orders': dashboard_data.get('recent_orders', [])
        },
        'timestamp': dashboard_data.get('timestamp', None)
    }
    
    print(json.dumps(result, indent=2))
    
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e),
        'data': None
    }
    print(json.dumps(error_result, indent=2))
    sys.exit(1) 