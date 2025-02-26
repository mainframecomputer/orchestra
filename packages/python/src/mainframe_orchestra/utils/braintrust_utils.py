# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

"""
Utility module for handling optional Braintrust functionality.
This module provides fallback decorators when Braintrust is not available.
"""

import os

# Check if Braintrust logging is explicitly disabled
BRAINTRUST_DISABLED = os.environ.get("BRAINTRUST_ORCHESTRA_DISABLED", "").lower() in ("true", "1", "yes")

try:
    from braintrust import traced, wrap_openai
    BRAINTRUST_AVAILABLE = True and not BRAINTRUST_DISABLED
except ImportError:
    BRAINTRUST_AVAILABLE = False
    def traced(type=None):
        """No-op decorator when Braintrust is not available"""
        def decorator(func):
            return func
        return decorator
    
    def wrap_openai(func):
        """No-op decorator when Braintrust is not available"""
        return func
