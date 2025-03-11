# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import requests
from typing import List, Dict, Optional
from ..utils.braintrust_utils import traced

class WikipediaTools:
    @traced(type="tool")
    @staticmethod
    def get_article(title, include_images=False):
        """
        Retrieve a Wikipedia article by its title.

        This method fetches the content of a Wikipedia article, including its extract, URL, and optionally, images.

        Args:
            title (str): The title of the Wikipedia article to retrieve.
            include_images (bool, optional): Whether to include images in the response. Defaults to False.

        Returns:
            dict: A dictionary containing the article data, including:
                - 'extract': The main text content of the article.
                - 'fullurl': The full URL of the article on Wikipedia.
                - 'pageid': The unique identifier of the page.
                - 'title': The title of the article.
                - 'thumbnail' (optional): Information about the article's thumbnail image, if available and requested.

        Raises:
            requests.exceptions.RequestException: If there's an error fetching the article from the Wikipedia API.
            KeyError, ValueError: If there's an error parsing the API response.

        Note:
            This method uses the Wikipedia API to fetch article data. The API has rate limits and usage policies
            that should be respected when making frequent requests.
        """
        print(f"Getting article for title: {title}")
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|info|pageimages",
            "inprop": "url",
            "redirects": "",
            "format": "json",
            "origin": "*",
            "pithumbsize": "400" if include_images else "0"
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            pages = data["query"]["pages"]
            page_id = next(iter(pages))
            article = pages[page_id]
            return article
        except requests.exceptions.RequestException as e:
            print(f"Error fetching article: {e}")
            return None
        except (KeyError, ValueError) as e:
            print(f"Error parsing response: {e}")
            return None

    @traced(type="tool")
    @staticmethod
    def search_articles(query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """
        Search for Wikipedia articles based on a given query.

        Args:
            query (str): The search query string.
            num_results (int, optional): The maximum number of search results to return. Defaults to 10.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing detailed information about each search result.
            Each dictionary includes:
                - 'title': The title of the article.
                - 'fullurl': The full URL of the article on Wikipedia.
                - 'snippet': A brief extract or snippet from the article.

        Raises:
            requests.exceptions.RequestException: If there's an error fetching search results from the Wikipedia API.
            KeyError, ValueError: If there's an error parsing the API response.

        Note:
            This method uses the Wikipedia API to perform the search. The API has rate limits and usage policies
            that should be respected when making frequent requests.
        """
        print(f"Searching articles for query: {query}")
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": num_results,
            "format": "json",
            "origin": "*"
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            search_results = data["query"]["search"]

            # Fetch additional details for each search result
            detailed_results = []
            for result in search_results:
                page_id = result['pageid']
                detailed_params = {
                    "action": "query",
                    "pageids": page_id,
                    "prop": "info|extracts|pageimages",
                    "inprop": "url",
                    "exintro": "",
                    "explaintext": "",
                    "pithumbsize": "250",
                    "format": "json",
                    "origin": "*"
                }
                detailed_response = requests.get(base_url, params=detailed_params)
                detailed_response.raise_for_status()
                detailed_data = detailed_response.json()
                page_data = detailed_data["query"]["pages"][str(page_id)]

                detailed_result = {
                    "title": page_data.get("title"),
                    "fullurl": page_data.get("fullurl"),
                    "snippet": page_data.get("extract", "")
                }
                detailed_results.append(detailed_result)

            return detailed_results
        except requests.exceptions.RequestException as e:
            print(f"Error searching articles: {e}")
            return []
        except (KeyError, ValueError) as e:
            print(f"Error parsing response: {e}")
            return []

    @traced(type="tool")
    @staticmethod
    def get_main_image(title: str, thumb_size: int = 250) -> Optional[str]:
        """
        Retrieve the main image for a given Wikipedia article title.

        This method queries the Wikipedia API to fetch the main image (thumbnail)
        associated with the specified article title.

        Args:
            title (str): The title of the Wikipedia article.
            thumb_size (int, optional): The desired size of the thumbnail in pixels. Defaults to 250.

        Returns:
            Optional[str]: The URL of the main image if found, None otherwise.

        Raises:
            requests.exceptions.RequestException: If there's an error in the HTTP request.
            KeyError, ValueError: If there's an error parsing the API response.
        """
        print(f"Getting main image for title: {title}")
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": title,
            "prop": "pageimages",
            "pithumbsize": thumb_size,
            "format": "json",
            "origin": "*"
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            pages = data["query"]["pages"]
            page_id = next(iter(pages))
            image_info = pages[page_id].get("thumbnail")
            if image_info:
                return image_info["source"]
            else:
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching main image: {e}")
            return None
        except (KeyError, ValueError) as e:
            print(f"Error parsing response: {e}")
            return None

    @traced(type="tool")
    @staticmethod
    def search_images(query: str, limit: int = 20, thumb_size: int = 250) -> List[Dict[str, str]]:
        """
        Search for images on Wikimedia Commons based on a given query.

        This method queries the Wikimedia Commons API to fetch images related to the specified query.

        Args:
            query (str): The search query for finding images.
            limit (int, optional): The maximum number of image results to return. Defaults to 20.
            thumb_size (int, optional): The desired size of the thumbnail in pixels. Defaults to 250.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing image information.
            Each dictionary includes 'title', 'url', and 'thumbnail' keys.

        Raises:
            requests.exceptions.RequestException: If there's an error in the HTTP request.
            KeyError, ValueError: If there's an error parsing the API response.
        """
        print(f"Searching images for query: {query}")
        base_url = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "generator": "search",
            "gsrnamespace": "6",
            "gsrsearch": f"intitle:{query}",
            "gsrlimit": limit,
            "prop": "pageimages|info",
            "pithumbsize": thumb_size,
            "inprop": "url",
            "format": "json",
            "origin": "*"
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            pages = data["query"]["pages"]
            image_results = []
            for page_id, page_data in pages.items():
                image_info = {
                    "title": page_data["title"],
                    "url": page_data["fullurl"],
                    "thumbnail": page_data.get("thumbnail", {}).get("source")
                }
                image_results.append(image_info)

            # New code to format the output
            formatted_results = []
            separator = "-" * 30  # Create a separator line
            for i, image in enumerate(image_results, 1):
                formatted_image = f"\nImage {i}:\n"
                formatted_image += f"  Title: {image['title']}\n"
                formatted_image += f"  URL: {image['url']}\n"
                formatted_image += f"  Thumbnail: {image['thumbnail']}\n"
                formatted_image += f"{separator}"
                formatted_results.append(formatted_image)

            return "".join(formatted_results)

        except requests.exceptions.RequestException as e:
            print(f"Error searching images: {e}")
            return "Error occurred while searching for images."
        except (KeyError, ValueError) as e:
            print(f"Error parsing response: {e}")
            return "Error occurred while parsing the response."
