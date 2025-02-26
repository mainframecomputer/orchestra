# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

"""
Utility module for handling optional Braintrust functionality.
This module provides fallback decorators when Braintrust is not available.
"""

import os

# Check if Braintrust integration is explicitly enabled or disabled via environment variable.
# Default to enabled if API key is present and no explicit setting
BRAINTRUST_API_KEY_EXISTS = os.environ.get("BRAINTRUST_API_KEY", "") != ""
BRAINTRUST_ENABLED = os.environ.get("BRAINTRUST_ORCHESTRA_ENABLED", "").lower() in ("true", "1", "yes") or (
    BRAINTRUST_API_KEY_EXISTS and os.environ.get("BRAINTRUST_ORCHESTRA_ENABLED", None) is None
)

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

# Try to import Braintrust if enabled
if BRAINTRUST_ENABLED:
    try:
        from braintrust import traced, wrap_openai
    except ImportError:
        # Keep the no-op implementations defined above
        pass
