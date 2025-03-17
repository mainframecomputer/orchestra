# Copyright 2025 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

def parse_json_response(response: str) -> dict:
    """
    An improved JSON parser that better handles text before JSON content.
    """
    import json
    import re

    # First attempt: Try to parse the entire response
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Second attempt: Find JSON by looking for the first { and matching closing }
        try:
            start_idx = response.find('{')
            if start_idx != -1:
                # Count opening and closing braces to find the complete JSON object
                open_count = 0
                for i in range(start_idx, len(response)):
                    if response[i] == '{':
                        open_count += 1
                    elif response[i] == '}':
                        open_count -= 1
                        if open_count == 0:
                            json_str = response[start_idx:i+1]
                            # Try to remove comments before parsing
                            try:
                                # Remove both single-line and multi-line comments
                                comment_pattern = r"//.*?(?:\n|$)|/\*.*?\*/"
                                cleaned_json = re.sub(comment_pattern, "", json_str, flags=re.DOTALL)
                                result = json.loads(cleaned_json)
                                if isinstance(result, dict):
                                    return result
                            except json.JSONDecodeError:
                                # If comment removal didn't work, try the original string
                                result = json.loads(json_str)
                                if isinstance(result, dict):
                                    return result
                            break
        except (json.JSONDecodeError, IndexError):
            pass

        # Third attempt: Use a more robust regex pattern
        json_pattern = r"(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})"
        json_matches = re.finditer(json_pattern, response, re.DOTALL)

        for match in json_matches:
            try:
                json_str = match.group(1)
                # Try to remove comments before parsing
                try:
                    # Remove both single-line and multi-line comments
                    comment_pattern = r"//.*?(?:\n|$)|/\*.*?\*/"
                    cleaned_json = re.sub(comment_pattern, "", json_str, flags=re.DOTALL)
                    result = json.loads(cleaned_json)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    # If comment removal didn't work, try the original string
                    result = json.loads(json_str)
                    if isinstance(result, dict):
                        return result
            except json.JSONDecodeError:
                continue

        # If all else fails, raise an error
        raise ValueError(f"Could not parse JSON from response: {response[:100]}...")
