# WebTools
The `WebTools` class provides a comprehensive set of static methods for interacting with various web services and APIs. These powerful tools enable developers to perform web searches, scrape websites, query academic databases, and retrieve weather information with ease and efficiency. By leveraging these methods, you can enhance your applications with rich, real-time data from across the internet.

## Class Methods Overview

1. `exa_search`: Perform a web search using the Exa Search API
2. `scrape_urls`: Scrape content from a list of URLs
3. `serper_search`: Perform a web search using the Serper API
4. `scrape_url_with_serper`: Scrape content from a URL using the Serper API
5. `query_arxiv_api`: Query the arXiv API for academic papers
6. `get_weather_data`: Retrieve weather data for a specified location

Let's dive into each method in detail, exploring their functionality, parameters, return values, and practical examples.

## exa_search

### Method Signature

```python
@staticmethod
def exa_search(queries: Union[str, List[str]], num_results: int = 10, search_type: str = "neural", num_sentences: int = 3, highlights_per_url: int = 3) -> dict:
```

### Parameters
- `queries` (Union[str, List[str]]): A single query string or a list of query strings to search for.
- `num_results` (int, optional): The number of search results to return per query. Default is 10.
- `search_type` (str, optional): The type of search to perform. Can be "neural" or "web". Default is "neural".
- `num_sentences` (int, optional): The number of sentences to include in the highlights for each result. Default is 3.
- `highlights_per_url` (int, optional): The number of highlights to include per URL. Default is 3.

### Return Value
- `dict`: A structured dictionary containing the search results for each query.

### Description
The `exa_search` method allows you to perform web searches using the Exa Search API. It takes one or more search queries and returns a structured dictionary containing the search results for each query.

The method first checks if the `EXA_API_KEY` environment variable is set. If not, it raises a `ValueError`. It then constructs the API endpoint URL and headers, including the API key.

For each query, the method constructs a payload dictionary containing the search parameters, such as the query string, search type, number of results, and highlight settings. It then sends a POST request to the API endpoint with the payload.

If the request is successful, the method restructures the response data into a more user-friendly format. It extracts the relevant information, such as the title, URL, author, and highlights for each search result, and appends it to the `structured_data` dictionary.

If an error occurs during the request or JSON decoding, the method prints an error message and retries the request up to 3 times with a 1-second delay between attempts. If all attempts fail, it adds an empty result for that query to the `structured_data` dictionary.

Finally, the method returns the `structured_data` dictionary containing the search results for each query.

### Example Usage

```python
queries = ["Python programming", "Machine learning algorithms"]
results = WebTools.exa_search(queries, num_results=5, search_type="neural", num_sentences=2, highlights_per_url=2)
print(results)
```

## scrape_urls

### Method Signature

```python
@staticmethod
def scrape_urls(urls: Union[str, List[str]], include_html: bool = False, include_links: bool = False) -> List[Dict[str, Union[str, List[str]]]]:
```

### Parameters
- `urls` (Union[str, List[str]]): A single URL string or a list of URL strings to scrape.
- `include_html` (bool, optional): Whether to include the HTML content of the scraped pages in the results. Default is False.
- `include_links` (bool, optional): Whether to include the links found on the scraped pages in the results. Default is False.

### Return Value
- `List[Dict[str, Union[str, List[str]]]]`: A list of dictionaries, where each dictionary represents a scraped URL and contains the URL, content, and optionally, the HTML content and links.

### Description
The `scrape_urls` method allows you to scrape content from one or more URLs. It takes a single URL string or a list of URL strings and returns a list of dictionaries containing the scraped data for each URL.

For each URL, the method sends a GET request with a random user agent header to avoid being blocked by websites. It adds a small delay between requests to avoid overwhelming the server.

If the request is successful, the method parses the HTML content using BeautifulSoup. It removes any script, style, and SVG tags from the parsed content.

The method then constructs a dictionary for each URL, containing the URL and the scraped content. If `include_html` is set to True, it includes the HTML content of the page in the dictionary. If `include_links` is set to True, it includes the links found on the page in the dictionary.

If an error occurs during the request or content processing, the method appends a dictionary to the results list with the URL and an error message describing the encountered issue.

Finally, the method returns the list of dictionaries containing the scraped data for each URL.

### Example Usage

```python
urls = ["https://www.example.com", "https://www.example.org"]
scraped_data = WebTools.scrape_urls(urls, include_html=True, include_links=True)
print(scraped_data)
```

## get_weather_data

### Method Signature

```python
@staticmethod
def get_weather_data(
    location: str,
    forecast_days: Optional[int] = None,
    include_current: bool = True,
    include_forecast: bool = True,
    include_astro: bool = False,
    include_hourly: bool = False,
    include_alerts: bool = False,
) -> str:
```

### Parameters
- `location` (str): The location to get the weather report for.
- `forecast_days` (Optional[int], optional): Number of days for the forecast (1-10). If provided, forecast is included. Default is None.
- `include_current` (bool, optional): Whether to include current conditions in the output. Default is True.
- `include_forecast` (bool, optional): Whether to include forecast data in the output. Default is True.
- `include_astro` (bool, optional): Whether to include astronomical data in the output. Default is False.
- `include_hourly` (bool, optional): Whether to include hourly forecast data in the output. Default is False.
- `include_alerts` (bool, optional): Whether to include weather alerts in the output. Default is False.

### Return Value
- `str`: Formatted string with the requested weather data.

### Raises
- `ValueError`: If the API key is not set, days are out of range, or if the API returns an error.
- `requests.RequestException`: If there's an error with the API request.

### Description
The `get_weather_data` method allows you to retrieve a detailed weather report for a specified location using the WeatherAPI service. It takes the location as a required parameter and several optional parameters to customize the output.

The method first checks if the `WEATHER_API_KEY` environment variable is set. If not, it raises a `ValueError`. It then constructs the API endpoint URL and parameters, including the API key, location, and other options based on the provided parameters.

If `forecast_days` is provided, it must be between 1 and 10. If `include_forecast` is True but `forecast_days` is not set, it defaults to 1 day.

The method sends a GET request to the WeatherAPI endpoint with the constructed parameters. If the request is successful, it retrieves the JSON response data. If the API returns an error, it raises a `ValueError` with the error message.

The method then formats the weather data into a readable string report. It includes the location details, current conditions (if `include_current` is True), forecast data (if `include_forecast` is True), astronomical data (if `include_astro` is True), hourly forecast data (if `include_hourly` is True), and weather alerts (if `include_alerts` is True).

Finally, the method returns the formatted weather report as a string.

### Example Usage

```python
location = "New York"
weather_report = WebTools.get_weather_data(location, forecast_days=3, include_current=True, include_forecast=True, include_astro=True, include_hourly=True, include_alerts=True)
print(weather_report)
```

These are just a few examples of the powerful methods available in the `WebTools` class. By leveraging these tools, you can easily integrate web search, web scraping, academic paper querying, and weather data retrieval into your applications, enabling you to build feature-rich and data-driven solutions.


