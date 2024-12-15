# Wikipedia Tools

The WikipediaTools class provides a set of static methods for interacting with the Wikipedia API, allowing you to retrieve article content, search for articles, and fetch images related to Wikipedia articles.

### Class Methods

##### get_article(title: str, include_images: bool = False)

Retrieves a Wikipedia article by its title. This method fetches the content of a Wikipedia article, including its extract, URL, and optionally, images.

```python
article = WikipediaTools.get_article("Python (programming language)", include_images=True)
```

##### search_articles(query: str, num_results: int = 10)

Searches for Wikipedia articles based on a given query. This method returns a list of dictionaries containing detailed information about each search result, including title, full URL, and a snippet from the article.

```python
search_results = WikipediaTools.search_articles("artificial intelligence", num_results=5)
```

##### get_main_image(title: str, thumb_size: int = 250)

Retrieves the main image for a given Wikipedia article title. This method returns the URL of the main image (thumbnail) associated with the specified article.

```python
main_image_url = WikipediaTools.get_main_image("Eiffel Tower", thumb_size=400)
```

##### search_images(query: str, limit: int = 20, thumb_size: int = 250)

Searches for images on Wikimedia Commons based on a given query. This method returns a list of dictionaries containing image information, including title, URL, and thumbnail URL.

```python
image_results = WikipediaTools.search_images("solar system", limit=10, thumb_size=300)
```

### Usage Notes

When using the WikipediaTools class, ensure that you have a stable internet connection to make API requests. The methods raise appropriate exceptions (requests.exceptions.RequestException) if the API requests fail, so make sure to handle them accordingly in your code.

The WikipediaTools class provides a set of static methods, which means you can directly call them using the class name without creating an instance of the class.

Remember to adhere to the Wikipedia API usage guidelines and rate limits when making requests. Excessive or abusive requests may result in temporary or permanent restrictions on your API access.

The get_article() method allows you to retrieve comprehensive information about a Wikipedia article, including its content and optionally, associated images. You can use this method to fetch detailed information about a specific topic.

The search_articles() method is useful for finding relevant Wikipedia articles based on a search query. It provides a list of search results with detailed information, allowing you to present users with a summary of matching articles.

Use the get_main_image() method when you need to retrieve the primary image associated with a Wikipedia article. This can be helpful for displaying visual content alongside article information.

The search_images() method allows you to find images related to a specific query on Wikimedia Commons. This is particularly useful when you need to retrieve multiple images related to a topic or concept.

When working with image-related methods (get_main_image() and search_images()), you can specify the desired thumbnail size to optimize image loading and display in your application.

All methods in the WikipediaTools class include error handling and logging. Make sure to implement appropriate error handling in your code to gracefully manage potential issues with API requests or response parsing.

