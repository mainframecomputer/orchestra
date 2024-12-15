# Writing Custom Tools

While the Orchestra framework provides a library of built-in tools, you may often need to create custom tools to extend its functionality or integrate specific libraries. This is one of the most powerful aspects of Orchestra because custom tools are what will allow your agents to perform tasks that are perfectly tailored to your needs.

This chapter will guide you through the process of writing custom tools that seamlessly integrate with Orchestra, focusing on clear documentation, proper error handling, and practical examples.

### Anatomy of a Custom Tool

A custom tool in Orchestra is typically implemented as a class with static methods, but can also be a class with instance methods for more complex operations. Each method or function represents a specific operation that can be used within Task objects. Here's a basic structure:

```python
import numpy as np
from typing import List, Union

class NumpyTools:
    @staticmethod
    def array_mean(arr: Union[List[float], np.ndarray]) -> Union[float, str]:
        """
        Calculate the mean of a given array.

        Args:
            arr (Union[List[float], np.ndarray]): Input array or list of numbers.

        Returns:
            Union[float, str]: The mean of the input array as a float, or an error message as a string.
        """
        try:
            arr = np.array(arr, dtype=float)
            if arr.size == 0:
                return "Error: Input array is empty."
            return float(np.mean(arr))
        except TypeError as e:
            return f"Error: Invalid input type. Expected a list or numpy array of numbers. Details: {e}"
        except Exception as e:
            return f"Error: An unexpected error occurred: {e}"
```

### Key Components of a Custom Tool

When creating a custom tool, focus on these essential elements:

- Clear and Descriptive Docstrings: Provide detailed information about the tool's purpose, input arguments, return values, and potential exceptions. **The docstring is shown to the LLM in its context window, so it's important to make it as clear and as detailed as possible for the LLM to understand and use it.** The more complex the args are and the tools are to use, the smarter the model may need to be to understand it. The LLM is only shown the docstring, not the code, so it's on you to write a good one! It can be tweaked and edited after to improve the robustness.

- Type Annotations: Use type hints to clearly specify the expected input types and return type of the tool. This helps the LLM understand the expected format and type of the data, and reduces the need for self-correction in the retry loop.

- Comprehensive Error Handling: Implement try-except blocks to catch and handle potential errors, returning informative error messages for the retry loop to assist the LLM in determining the correct input. You can make unfixable (non-user caused) errors break the flow for you to debug, and any user-caused errors like fixable inputs can be handled with a Return statement with a helpful message on how to fix the input. The more complex your error catching and reporting is, the more likely it is that the LLM will be able to correct the input on the next try.

- Input Validation: Validate input arguments to ensure they meet the tool's requirements before processing. In other words, call the tool yourself with the correct args and incorrect args to see if it aligns with expected outcomes before handing it over to the LLM to use.

### Requirements for Custom Tools

- The tool must be a function with informative (but not overwhelming) docstrings.
- The tool must be assigned to an agent and passed to the task in a set like this: `tools = {ClassName.method_name}`.
- The tool should be easy to understand and use. The more flexible the inputs are, the fewer tools you will need to create. That said, focused tools will aid in usability and robustness with a diversity of model sizes.

When designing tools, it's important to balance complexity with usability. While more advanced language models can handle diverse inputs and complex functionalities, simpler tools are often more effective, especially when using less powerful LLMs. Consider breaking down complex tools into multiple, more focused ones (e.g., 'search_news' and 'search_articles' instead of a single, parameter-heavy search function). This approach not only aids in tool selection but also enhances usability across different LLM capabilities and cost tiers. More granular tools are easier to compose, reconfigure and reuse across different agents and tasks.

### Advanced Error Handling for Retry Loops

To support Orchestra's retry mechanism, it's crucial to implement error handling that provides detailed and actionable error messages. Here's an example of how to structure error handling for effective retries:

```python
class AdvancedNumpyTools:
    @staticmethod
    def array_operation(arr: Union[List[float], np.ndarray], operation: str) -> Union[float, str]:
        """
        Perform a specified operation on the input array.

        Args:
            arr (Union[List[float], np.ndarray]): Input array or list of numbers.
            operation (str): The operation to perform ('mean', 'median', or 'std').

        Returns:
            Union[float, str]: The result of the specified operation as a float, or an error message as a string.
        """
        try:
            arr = np.array(arr, dtype=float)
            if arr.size == 0:
                error_msg = "Error: Input array is empty. Please provide a non-empty array of numbers."
                print(error_msg)
                return error_msg
            
            if operation == 'mean':
                return float(np.mean(arr))
            elif operation == 'median':
                return float(np.median(arr))
            elif operation == 'std':
                return float(np.std(arr))
            else:
                error_msg = f"Error: Invalid operation '{operation}'. Supported operations are 'mean', 'median', and 'std'."
                print(error_msg)
                return error_msg
        except TypeError as e:
            error_msg = f"Error: Invalid input type: {e}. Please provide a list or numpy array of numbers."
            print(error_msg)
            return error_msg
        except ValueError as e:
            error_msg = f"Error: Invalid value in input array: {e}. Please ensure all elements are numeric."
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error: An unexpected error occurred during {operation} calculation: {e}"
            print(error_msg)
            raise Exception(error_msg)  # Raise an exception to break the flow and allow for debugging
```

In this example, the error messages are designed to provide clear guidance on how to correct the input or usage of the tool, facilitating effective retries in Orchestra.

### Allowing Agents to Interact with External APIs

To demonstrate how to integrate a new API with your agents using custom tools, let's use the popular OpenWeatherMap API as an example. This API provides weather data for locations worldwide. We'll create a custom tool that wraps some of its endpoints into functions that can be used by AI agents.

First, you'll need to sign up for an API key at [OpenWeatherMap](https://openweathermap.org/api).

Then, you can read through their API documentation to research their available endpoints and parameters and consider which would be useful to your agents and general flow of tasks and use cases.

Here's an example of how you might create a custom tool for the OpenWeatherMap API:

```python
import requests
import os
from dotenv import load_dotenv

class OpenWeatherMapTools:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('OPENWEATHERMAP_API_KEY')
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key not found. Please set the OPENWEATHERMAP_API_KEY environment variable.")
        self.base_url = "https://api.openweathermap.org/data/2.5"

    def _make_request(self, endpoint, params):
        """
        Helper method to make API requests with error handling. Not for LLMs.
        """
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return f"Error making API request: {str(e)}"

    def get_current_weather(self, city):
        """
        Get current weather data for a specific city.

        Args:
            city (str): The name of the city to get weather data for.

        Returns:
            dict: JSON response containing current weather data, or an error message.
        """
        endpoint = f"{self.base_url}/weather"
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric"
        }
        return self._make_request(endpoint, params)

    def get_forecast(self, city, days=5):
        """
        Get weather forecast for a specific city.

        Args:
            city (str): The name of the city to get weather forecast for.
            days (int): The number of days to get the forecast for. Defaults to 5.

        Returns:
            dict: JSON response containing weather forecast data, or an error message.
        """
        endpoint = f"{self.base_url}/forecast"
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric",
            "cnt": days * 8  # 3-hour steps, 8 per day
        }
        return self._make_request(endpoint, params)

    def get_air_pollution(self, lat, lon):
        """
        Get current air pollution data for specific coordinates.
        """
        endpoint = f"{self.base_url}/air_pollution"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key
        }
        return self._make_request(endpoint, params)
```

In this example, we've created a custom tool class `OpenWeatherMapTools` with methods for fetching current weather, forecast, and air pollution data for a given city or coordinates. 

To assign them to a weather agent, you would pass them to the agent like this: 

```python
from mainframe_orchestra import Agent, OpenrouterModels
weather_reporter = Agent(
    role="Weather Reporter",
    goal="Report on weather conditions for a given city",
    attributes="knowledgeable, precise, helpful",
    tools={OpenWeatherMapTools.get_current_weather, OpenWeatherMapTools.get_forecast, OpenWeatherMapTools.get_air_pollution},
    llm=OpenrouterModels.haiku
)
```

Then, this agent can use any of these tools within their tasks to report on the weather conditions for different cities.

### Conclusion

Creating custom tools is an essential skill for extending the capabilities of Orchestra to meet your specific needs. By following the guidelines and best practices outlined in this chapter, you can develop robust, well-documented, and easily integrable custom tools that enhance the power and flexibility of your AI-driven workflows.

