# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

"""
Utility module for handling optional Braintrust functionality.
This module provides fallback decorators when Braintrust is not available.
"""

import os

# Check if Braintrust integration is explicitly disabled via environment variable.
BRAINTRUST_DISABLED = os.environ.get("BRAINTRUST_ORCHESTRA_DISABLED", "").lower() in ("true", "1", "yes")

# Default implementation of no-op decorators
def traced(func=None, **kwargs):
    """No-op decorator when Braintrust is not available"""
    if func is None:
        def decorator(f):
            return f
        return decorator
    return func

def wrap_openai(func):
    """No-op decorator when Braintrust is not available"""
    return func

# Try to import Braintrust if not disabled
if not BRAINTRUST_DISABLED:
    try:
        from braintrust import traced, wrap_openai
    except ImportError:
        # Keep the no-op implementations defined above
        pass
