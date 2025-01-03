from mainframe_orchestra import Task, Agent, OpenaiModels, Conduct, WebTools, set_verbosity
from typing import Union, Optional, List
import requests
import json
import os
from dotenv import load_dotenv

# This simple example demonstrates how to use the Orchestra framework to create a team of agents to search for products based on an image URL.

# Define the custom search tools class
class SearchTools:
    load_dotenv()

    def search_shopping(
            query: Union[str, List[str]],
            num_results: Optional[int] = range(3, 10),
            min_price: Optional[float] = None,
            max_price: Optional[float] = None
        ) -> Union[str, List[str]]:
            """
            Perform a shopping search using the Serper API and format the results.

            Args:
                query (Union[str, List[str]]): The search query or a list of search queries.
                num_results (Optional[int]): Number of results to return (default range: 3-10).
                min_price (Optional[float]): Minimum price filter for products.
                max_price (Optional[float]): Maximum price filter for products.

            Returns:
                Union[str, List[str]]: A formatted string or list of formatted strings containing 
                the shopping search results.

            Raises:
                ValueError: If the API key is not set or if there's an error with the API call.
                requests.RequestException: If there's an error with the HTTP request.
            """
            # Get API key from environment variable
            api_key = os.getenv("SERPER_API_KEY")
            if not api_key:
                raise ValueError("SERPER_API_KEY environment variable is not set")

            BASE_URL = "https://google.serper.dev/shopping"
            queries = [query] if isinstance(query, str) else query
            results_list = []

            for single_query in queries:
                payload = {
                    "q": single_query,
                    "gl": "us",
                    "hl": "en",
                }

                if num_results is not None:
                    payload["num"] = num_results

                # Add price range filters if provided
                if min_price is not None or max_price is not None:
                    price_range = ""
                    if min_price is not None:
                        price_range += f"{min_price}"
                    price_range += ".."
                    if max_price is not None:
                        price_range += f"{max_price}"
                    payload["price"] = price_range

                headers = {
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json"
                }

                try:
                    response = requests.post(BASE_URL, headers=headers, data=json.dumps(payload))
                    response.raise_for_status()
                    results = response.json()
                    
                    formatted_results = "Shopping Results:\n"

                    if "shopping" in results:
                        for i, product in enumerate(results["shopping"], 1):
                            formatted_results += f"{i}. {product.get('title', 'No Title')}\n"
                            formatted_results += f"   Price: {product.get('price', 'Price not available')}\n"
                            formatted_results += f"   Rating: {product.get('rating', 'No rating')} "
                            if "ratingCount" in product:
                                formatted_results += f"({product['ratingCount']} reviews)\n"
                            else:
                                formatted_results += "\n"
                            formatted_results += f"   Seller: {product.get('seller', 'Seller not available')}\n"
                            formatted_results += f"   URL: {product.get('link', 'No Link')}\n"
                            if "imageUrl" in product:
                                formatted_results += f"   Image URL: {product['imageUrl']}\n"
                            formatted_results += "\n"

                    results_list.append(formatted_results.strip())
                
                except requests.RequestException as e:
                    results_list.append(f"Error making request to Serper API for query '{single_query}': {str(e)}")
                except json.JSONDecodeError:
                    results_list.append(f"Error decoding JSON response from Serper API for query '{single_query}'")

            return results_list[0] if len(results_list) == 1 else results_list

    def search_lens(
            image_url: str,
            num_results: int = 10
        ) -> str:
            """
            Perform a reverse image search using the Serper API Lens feature.

            Args:
                image_url (str): The URL of the image to search for.
                num_results (int): Number of results to return (default: 10).

            Returns:
                str: A formatted string containing the image search results.

            Raises:
                ValueError: If the API key is not set or if there's an error with the API call.
                requests.RequestException: If there's an error with the HTTP request.
            """
            # Get API key from environment variable
            api_key = os.getenv("SERPER_API_KEY")
            if not api_key:
                raise ValueError("SERPER_API_KEY environment variable is not set")

            BASE_URL = "https://google.serper.dev/lens"

            payload = {
                "url": image_url,
                "gl": "us",
                "hl": "en",
            }

            if num_results is not None:
                payload["num"] = num_results

            headers = {
                "X-API-KEY": api_key,
                "Content-Type": "application/json"
            }

            try:
                response = requests.post(BASE_URL, headers=headers, data=json.dumps(payload))
                response.raise_for_status()
                results = response.json()
                
                formatted_results = "Image Search (Lens) Results:\n"

                if "organic" in results:
                    for i, result in enumerate(results["organic"], 1):
                        formatted_results += f"{i}. {result.get('title', 'No Title')}\n"
                        formatted_results += f"   Source: {result.get('source', 'No Source')}\n"
                        formatted_results += f"   URL: {result.get('link', 'No Link')}\n"
                        if "imageUrl" in result:
                            formatted_results += f"   Similar Image: {result['imageUrl']}\n"
                        formatted_results += "\n"

                return formatted_results.strip()
            
            except requests.RequestException as e:
                return f"Error making request to Serper API Lens: {str(e)}"
            except json.JSONDecodeError:
                return "Error decoding JSON response from Serper API Lens"

# Set verbosity level
set_verbosity(0)

# Define the team of agents
image_analyst = Agent(
    agent_id="image_analyst",
    role="Image Search Specialist",
    goal="Analyze images using reverse image search and extract key product details",
    attributes="You have expertise in image analysis and product identification. You should identify the main product in the image and create appropriate search queries.",
    llm=OpenaiModels.gpt_4o_mini,
    tools={SearchTools.search_lens}
)

shopping_analyst = Agent(
    agent_id="shopping_analyst",
    role="Shopping Search Specialist",
    goal="Find the best shopping results based on product descriptions and scrape the websites listed in the results to summarize the product information",
    attributes="You have expertise in product search and comparison. You know to search for similar products and check out the websites listed in the results.",
    llm=OpenaiModels.gpt_4o_mini,
    tools={SearchTools.search_shopping, WebTools.scrape_url_with_serper}
)

info_researcher = Agent(
    agent_id="info_researcher",
    role="Product Information Researcher",
    goal="Research detailed product information and reviews from web sources",
    attributes="You have expertise in gathering and analyzing product information from various web sources. When given URLs or product details, you should research and summarize key information, reviews, and relevant details. You write comprehensive reports in natural language, avoiding lists where possible.",
    llm=OpenaiModels.gpt_4o_mini,
    tools={WebTools.exa_search, WebTools.scrape_url_with_serper}
)

conductor = Agent(
    agent_id="conductor",
    role="Search Coordinator",
    goal="Coordinate the team to provide comprehensive product information from images",
    attributes="""You are an expert coordinator who knows how to get the best results from your specialized team. You avoid lists where possible, preferring to write natural language and comprehensive reports expounding the results of your team's research.""",
    llm=OpenaiModels.gpt_4o,
    tools=[Conduct.conduct_tool(image_analyst, shopping_analyst, info_researcher)]
)

def chat_task(userinput):
    return Task.create(
        agent=conductor,
        instruction=f"Orchestrate your team to provide comprehensive product information from image URLs. Provide the URL in whole in the instructions. Begin by having the image_analyst examine the image thoroughly to identify the product and its key characteristics, then direct the shopping_analyst to search for similar items across different retailers and price points and scrape the websites listed in the results to summarize the product information. Finally, task the info_researcher with gathering deeper product information from across the web. They should focus on finding reviews, specifications, and expert opinions that will help the user make an informed decision. Be sure to request the URLs for the shopping results and product information for user to read after the report. Be extensive and exhaustive in your final report. \n\nImage url to analyze: {userinput}"
    )

def main():
    print("Welcome! Please provide an image URL to search for similar products.")
    print("Type 'quit' to exit.")
    userinput = input("\nEnter image URL: ")
    response = chat_task(userinput)
    print(f"\nResults:\n{response}")

if __name__ == "__main__":
    main()