# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

from datetime import datetime
import base64
import re
import json
from typing import Union


class Utils:
    @staticmethod
    def update_conversation_history(history, role, content):
        """
        Format a message with a timestamp and role, then update the conversation history.

        Args:
            history (list): The current conversation history as a list of formatted messages.
            role (str): The role of the message sender (either "User" or "Assistant").
            content (str): The message content.

        Returns:
            list: Updated conversation history.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {role}: {content}"
        history.append(formatted_message)
        return history

    @staticmethod
    def image_to_base64(image_path: str, scale_factor: float = 0.5) -> str:
        """
        Convert an image to a base64-encoded string, with optional resizing.

        Args:
            image_path (str): The path to the image file.
            scale_factor (float, optional): Factor to scale the image dimensions. Defaults to 0.5.

        Returns:
            str: Base64-encoded string representation of the (optionally resized) image.

        Raises:
            IOError: If there's an error opening or processing the image file.
        """
        import numpy as np
        from io import BytesIO

        with open(image_path, "rb") as image_file:
            img = np.frombuffer(image_file.read(), dtype=np.uint8)
            img = img.reshape((-1, 3))  # Assuming 3 channels (RGB)
            resized_img = img[:: int(1 / scale_factor)]
            buffer = BytesIO()
            np.save(buffer, resized_img, allow_pickle=False)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def parse_json_response(response: Union[str, dict]) -> dict:
        """
        Parse a JSON object from a string response or return the dict if already parsed.

        Args:
            response (Union[str, dict]): The response, either a string containing a JSON object or an already parsed dict.

        Returns:
            dict: The parsed JSON object.

        Raises:
            ValueError: If no valid JSON object is found in the response string.
        """
        if isinstance(response, dict):
            return response

        if isinstance(response, str):
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                json_match = re.search(r"\{.*?\}", response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        raise ValueError(f"Failed to parse JSON from response: {response}")
                else:
                    raise ValueError(f"No JSON object found in response: {response}")

        raise ValueError(f"Unexpected response type: {type(response)}")
