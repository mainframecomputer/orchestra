# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import os
import json
from typing import Any, List, Dict, Union, Literal, Optional
from bs4 import BeautifulSoup
from ..utils.braintrust_utils import traced
import requests
import time
import random
from fake_useragent import UserAgent
from dotenv import load_dotenv

class WebTools:
    @traced(type="tool")
    @staticmethod
    def exa_search(queries: Union[str, List[str]], num_results: int = 10, search_type: str = "neural", num_sentences: int = 5, highlights_per_url: int = 3) -> dict:
        """
        Searches the internet using the Exa search engine and returns a structured response.

        This function sends one or more search queries to the Exa API, processes the responses,
        and returns a structured dictionary containing the search results.

        Args:
            queries (Union[str, List[str]]): The search query string or a list of query strings.
            num_results (int, optional): The number of search results to return per query. Defaults to 3.
            search_type (str, optional): The type of search to perform. Can be 'neural' or 'keyword'. Defaults to 'neural'.
            num_sentences (int, optional): The number of sentences to include in each highlight. Defaults to 3.
            highlights_per_url (int, optional): The number of highlights to include per URL. Defaults to 3.

        Returns:
            dict: A structured dictionary containing the search results. The dictionary includes:
                - 'queries': The original search query or queries.
                - 'results': A list of dictionaries, each containing:
                    - 'query': The specific query for this result set.
                    - 'data': A list of dictionaries, each containing:
                        - 'title': The title of the search result.
                        - 'url': The URL of the search result.
                        - 'author': The author of the content (if available).
                        - 'highlights': A string of highlighted text snippets from the search result.

        Raises:
            requests.exceptions.RequestException: If there's an error with the API request.
            json.JSONDecodeError: If the API response cannot be parsed as JSON.

        Note:
            This function requires the EXA_API_KEY environment variable to be set.
        """
        print(f"Searching Exa for: '{queries}'")

        # Load environment variables from .env file
        load_dotenv()

        # Retrieve the API key from environment variables
        api_key = os.getenv("EXA_API_KEY")
        if not api_key:
            raise ValueError("EXA_API_KEY environment variable is not set")

        # Convert single query to list for consistent processing
        if isinstance(queries, str):
            queries = [queries]

        # Define the API endpoint and headers
        url = "https://api.exa.ai/search"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": api_key  # Use the retrieved API key
        }

        structured_data = {
            "queries": queries,
            "results": []
        }

        for query in queries:
            # Define the payload
            payload = {
                "query": query,
                "type": search_type,
                "use_autoprompt": False,
                "num_results": num_results,
                "contents": {
                    "highlights": {
                        "numSentences": num_sentences,
                        "highlightsPerUrl": highlights_per_url,
                        "query": query
                    }
                }
            }

            # Attempt to make the POST request with retries
            for attempt in range(3):  # Retry up to 3 times
                try:
                    response = requests.post(url, headers=headers, json=payload)
                    response.raise_for_status()  # Will raise an exception for HTTP errors
                    # Print the raw response
                    #print(f"DEBUG: Raw API response for query '{query}': {response.text}")
                    # Restructure and clean up the response data
                    data = response.json()
                    query_results = {
                        "query": query,
                        "data": []
                    }
                    for result in data["results"]:
                        structured_result = {
                            "title": result["title"],
                            "url": result["url"],
                            "author": result["author"],
                            "highlights": "\n".join(result["highlights"])
                        }
                        query_results["data"].append(structured_result)

                    structured_data["results"].append(query_results)
                    break  # Success, break the retry loop

                except requests.exceptions.HTTPError as e:
                    print(f"HTTP error occurred for query '{query}': {e}")
                except requests.exceptions.RequestException as e:
                    print(f"Error during request to Exa for query '{query}': {e}")
                except ValueError as e:
                    print(f"Error decoding JSON for query '{query}': {e}")

                if attempt < 2:  # Don't sleep after the last attempt
                    print(f"Retrying query '{query}'...")
                    time.sleep(1)  # Wait for 1 second before retrying

            else:  # This else clause is executed if the for loop completes without breaking
                print(f"All attempts failed for query '{query}'. Adding empty result.")
                structured_data["results"].append({"query": query, "data": []})

        return structured_data

    @traced(type="tool")
    @staticmethod
    def scrape_urls(urls: Union[str, List[str]], include_html: bool = False, include_links: bool = False) -> List[Dict[str, Any]]:
        """
        Scrape one or more webpages and return the content.

        Args:
            urls (Union[str, List[str]]): A single URL string or a list of URL strings to scrape.
            include_html (bool, optional): Whether to include the HTML content of the page. Defaults to False.
            include_links (bool, optional): Whether to include the links on the page. Defaults to False.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the URL and the scraped content.
        """
        if isinstance(urls, str):
            urls = [urls]

        if not isinstance(include_html, bool) or not isinstance(include_links, bool):
            raise ValueError("include_html and include_links must be boolean values")

        results = []
        ua = UserAgent()

        for url in urls:
            try:
                time.sleep(random.uniform(.1, .3))
                headers = {'User-Agent': ua.random}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                response.encoding = response.apparent_encoding

                soup = BeautifulSoup(response.text, 'html.parser')

                # Remove common non-content elements
                for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    element.decompose()

                # Try to find the main content
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')

                if main_content:
                    content_soup = main_content
                else:
                    # If no main content found, use the whole body but remove potential sidebars
                    content_soup = soup.find('body')
                    if content_soup:
                        for sidebar in content_soup(['aside', 'div'], class_=['sidebar', 'widget']):
                            sidebar.decompose()

                result: Dict[str, Any] = {'url': url}

                if include_html:
                    result['content'] = str(content_soup) if content_soup else ''
                else:
                    result['content'] = content_soup.get_text(separator=' ', strip=True) if content_soup else ''

                if include_links:
                    result['links'] = [a['href'] for a in (content_soup.find_all('a', href=True) if content_soup else [])]

                results.append(result)

            except requests.RequestException as e:
                results.append({'url': url, 'error': f"Error fetching {url}: {str(e)}"})
            except UnicodeDecodeError as e:
                results.append({'url': url, 'error': f"Encoding error for {url}: {str(e)}"})
            except (AttributeError, ValueError, TypeError) as e:
                results.append({'url': url, 'error': f"Error processing {url}: {str(e)}"})

        return results

    @traced(type="tool")
    @staticmethod
    def serper_search(
        query: Union[str, List[str]],
        search_type: Literal["search", "news", "images", "shopping"] = "search",
        num_results: Optional[int] = range(3, 10),
        date_range: Optional[Literal["h", "d", "w", "m", "y"]] = None,
        location: Optional[str] = None
    ) -> Union[str, List[str]]:
        """
        Perform a search using the Serper API and format the results.

        Args:
            query (Union[str, List[str]]): The search query or a list of search queries.
            search_type (Literal["search", "news", "images", "shopping"]): The type of search to perform.
            num_results (Optional[int]): Number of results to return (default range: 3-10).
            date_range (Optional[Literal["h", "d", "w", "m", "y"]]): Date range for results (h: hour, d: day, w: week, m: month, y: year).
            location (Optional[str]): Specific location for the search.

        Returns:
            Union[str, List[str]]: A formatted string or list of formatted strings containing the search results.

        Raises:
            ValueError: If the API key is not set or if there's an error with the API call.
            requests.RequestException: If there's an error with the HTTP request.
        """
        # Load environment variables
        load_dotenv()

        # Get API key from environment variable
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError("SERPER_API_KEY environment variable is not set")

        # Define the base URL
        BASE_URL = "https://google.serper.dev"

        # Construct the URL based on the search type
        url = f"{BASE_URL}/{search_type}"

        # Convert query to list if it's a string
        queries = [query] if isinstance(query, str) else query

        results_list = []

        for single_query in queries:
            # Prepare the payload
            payload = {
                "q": single_query,
                "gl": "us",
                "hl": "en",
            }

            # Add num_results to payload if provided
            if num_results is not None:
                payload["num"] = num_results

            # Add optional parameters if provided
            if date_range:
                payload["tbs"] = f"qdr:{date_range}"
            if location:
                payload["location"] = location

            # Prepare headers
            headers = {
                "X-API-KEY": api_key,
                "Content-Type": "application/json"
            }

            try:
                # Make the API call
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                response.raise_for_status()  # Raise an exception for HTTP errors

                # Parse the JSON response
                results = response.json()

                # Format the results
                formatted_results = ""

                if "organic" in results:
                    formatted_results += "Organic Results:\n"
                    for i, result in enumerate(results["organic"], 1):
                        formatted_results += f"{i}. {result.get('title', 'No Title')}\n"
                        formatted_results += f"   URL: {result.get('link', 'No Link')}\n"
                        formatted_results += f"   Snippet: {result.get('snippet', 'No Snippet')}\n\n"

                if "news" in results:
                    formatted_results += "News Results:\n"
                    for i, news in enumerate(results["news"], 1):
                        formatted_results += f"{i}. {news.get('title', 'No Title')}\n"
                        formatted_results += f"   Source: {news.get('source', 'No Source')}\n"
                        formatted_results += f"   URL: {news.get('link', 'No Link')}\n"
                        formatted_results += f"   Date: {news.get('date', 'No Date')}\n"
                        formatted_results += f"   Snippet: {news.get('snippet', 'No Snippet')}\n"
                        formatted_results += f"   Image URL: {news.get('imageUrl', 'No Image URL')}\n\n"

                if "images" in results:
                    formatted_results += "Image Results:\n"
                    for i, image in enumerate(results["images"], 1):
                        formatted_results += f"{i}. {image.get('title', 'No Title')}\n"
                        formatted_results += f"   URL: {image.get('link', 'No Link')}\n"
                        formatted_results += f"   Source: {image.get('source', 'No Source')}\n\n"

                if "shopping" in results:
                    formatted_results += "Shopping Results:\n"
                    for i, item in enumerate(results["shopping"], 1):
                        formatted_results += f"{i}. {item.get('title', 'No Title')}\n"
                        formatted_results += f"   Price: {item.get('price', 'No Price')}\n"
                        formatted_results += f"   URL: {item.get('link', 'No Link')}\n\n"

                results_list.append(formatted_results.strip())

            except requests.RequestException as e:
                results_list.append(f"Error making request to Serper API for query '{single_query}': {str(e)}")
            except json.JSONDecodeError:
                results_list.append(f"Error decoding JSON response from Serper API for query '{single_query}'")

        return results_list[0] if len(results_list) == 1 else results_list

    @traced(type="tool")
    @staticmethod
    def scrape_url_with_serper(urls: Union[str, List[str]]) -> Union[dict, List[dict]]:
        """
        Scrape one or more webpages using the Serper API.

        Args:
            urls (Union[str, List[str]]): A single URL string or a list of URL strings to scrape.

        Returns:
            Union[dict, List[dict]]: A single JSON response or a list of JSON responses from the Serper API.

        Raises:
            ValueError: If the API key is not set or if the input is neither a string nor a list.
            requests.RequestException: If there's an error with the API request.
            json.JSONDecodeError: If the API response cannot be parsed as JSON.
        """
        #print(f"Attempting to scrape URL(s): {urls}")

        # Load environment variables
        load_dotenv()

        # Get API key from environment variable
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError("SERPER_API_KEY environment variable is not set")

        serper_url = "https://scrape.serper.dev"
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }

        def scrape_single_url(url: str) -> dict:
            payload = json.dumps({"url": url})
            try:
                response = requests.post(serper_url, headers=headers, data=payload)
                response.raise_for_status()
                scraped_data = response.json()
                print(f"Successfully scraped URL: {url}")
                return scraped_data
            except requests.RequestException as e:
                print(f"Error making request to Serper API for URL {url}: {str(e)}")
                raise
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON response from Serper API for URL {url}: {str(e)}")
                raise

        if isinstance(urls, str):
            return scrape_single_url(urls)
        elif isinstance(urls, list):
            return [scrape_single_url(url) for url in urls]
        else:
            raise ValueError("Input must be a string (single URL) or a list of strings (multiple URLs)")

    @traced(type="tool")
    @staticmethod
    def query_arxiv_api(
        query: str,
        max_results: int = 10,
        sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "relevance",
        include_abstract: bool = True
    ) -> dict:
        """
        Query the Arxiv API and return simplified results for each entry.

        Args:
            query (str): Search query string.
            max_results (int): The maximum number of results to return. Default is 10.
            sort_by (Literal["relevance", "lastUpdatedDate", "submittedDate"]): The sorting criteria. Default is "relevance".
            include_abstract (bool): Whether to include the abstract in the response. Default is True.

        Returns:
            dict: A structured dictionary containing the simplified query results.

        Raises:
            ValueError: If there's an error with the API call or response parsing.
            requests.RequestException: If there's an error with the HTTP request.
        """
        base_url = "http://export.arxiv.org/api/query"

        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending"
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            feed = BeautifulSoup(response.text, "lxml-xml")
            entries = feed.find_all("entry")

            results = {
                "total_results": int(feed.find("opensearch:totalResults").text),
                "entries": []
            }

            for entry in entries:
                entry_data = {
                    "id": entry.find("id").text,
                    "title": entry.find("title").text,
                    "authors": [author.find("name").text for author in entry.find_all("author")],
                    "published": entry.find("published").text,
                    "updated": entry.find("updated").text,
                    "categories": [category["term"] for category in entry.find_all("category")],
                    "primary_category": entry.find("arxiv:primary_category")["term"],
                    "link": entry.find("link", attrs={"type": "text/html"})["href"]
                }

                if include_abstract:
                    entry_data["abstract"] = entry.find("summary").text

                results["entries"].append(entry_data)

            return results

        except (ValueError, requests.exceptions.RequestException) as e:
            print(f"Error querying Arxiv API: {e}")
            raise

    @traced(type="tool")
    @staticmethod
    def get_weather_data(
        location: str,
        forecast_days: Optional[int] = None,
        include_current: bool = True,
        include_forecast: bool = True,  # New parameter
        include_astro: bool = False,
        include_hourly: bool = False,
        include_alerts: bool = False,
    ) -> str:
        """
        Get a detailed weather report for a location in a readable format.

        Args:
            location (str): The location to get the weather report for.
            forecast_days (Optional[int]): Number of days for the forecast (1-10). If provided, forecast is included.
            include_current (bool): Whether to include current conditions in the output.
            include_forecast (bool): Whether to include forecast data in the output.
            include_astro (bool): Whether to include astronomical data in the output.
            include_hourly (bool): Whether to include hourly forecast data in the output.
            include_alerts (bool): Whether to include weather alerts in the output.

        Returns:
            str: Formatted string with the requested weather data.

        Raises:
            ValueError: If the API key is not set, days are out of range, or if the API returns an error.
            requests.RequestException: If there's an error with the API request.
        """
        load_dotenv()
        api_key = os.getenv("WEATHER_API_KEY")
        if not api_key:
            raise ValueError("WEATHER_API_KEY environment variable is not set")

        BASE_URL = "http://api.weatherapi.com/v1"
        endpoint = 'forecast.json'

        params = {
            'key': api_key,
            'q': location,
            'aqi': 'yes',
            'alerts': 'yes',
        }

        if forecast_days is not None:
            if not 1 <= forecast_days <= 10:
                return f"Error: forecast_days must be between 1 and 10. You provided: {forecast_days}"
            params['forecast_days'] = forecast_days
        elif include_forecast:
            params['forecast_days'] = 1

        try:
            response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
            response.raise_for_status()
            data = response.json()

            if 'error' in data:
                raise ValueError(f"API Error: {data['error']['message']}")
        except requests.RequestException as e:
            raise requests.RequestException(f"Error making request to WeatherAPI: {str(e)}")

        report = "Forecast:\n"
        for key, value in data['location'].items():
            report += f"{key}: {value}\n"

        if include_current:
            report += "\nCurrent Conditions:\n"
            for key, value in data['current'].items():
                if isinstance(value, dict):
                    report += f"{key}:\n"
                    for sub_key, sub_value in value.items():
                        report += f"  {sub_key}: {sub_value}\n"
                else:
                    report += f"{key}: {value}\n"

            if include_astro:
                report += "Current Astro:\n"
                for key, value in data['forecast']['forecastday'][0]['astro'].items():
                    report += f"  {key}: {value}\n"

        if include_forecast and 'forecast' in data:
            report += "\nForecast:\n"
            for day in data['forecast']['forecastday']:
                report += f"\nDate: {day['date']}\n"
                for key, value in day['day'].items():
                    if isinstance(value, dict):
                        report += f"{key}:\n"
                        for sub_key, sub_value in value.items():
                            report += f"  {sub_key}: {sub_value}\n"
                    else:
                        report += f"{key}: {value}\n"

                if include_astro:
                    report += "Astro:\n"
                    for key, value in day['astro'].items():
                        report += f"  {key}: {value}\n"

                if include_hourly:
                    report += "Hour by hour:\n"
                    for hour in day['hour']:
                        report += f"  Time: {hour['time']}\n"
                        for key, value in hour.items():
                            if key != 'time':
                                if isinstance(value, dict):
                                    report += f"    {key}:\n"
                                    for sub_key, sub_value in value.items():
                                        report += f"      {sub_key}: {sub_value}\n"
                                else:
                                    report += f"    {key}: {value}\n"

        if include_alerts and 'alerts' in data and 'alert' in data['alerts']:
            report += "\nWeather Alerts:\n"
            for alert in data['alerts']['alert']:
                for key, value in alert.items():
                    report += f"{key}: {value}\n"
                report += "\n"

        return report

