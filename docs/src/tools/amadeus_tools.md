# Amadeus Tools

The AmadeusTools class provides a comprehensive set of methods to interact with the Amadeus API for flight-related operations. It offers powerful functionality for searching flights, finding the cheapest travel dates, and getting flight inspiration. This class is designed to simplify the process of working with the Amadeus API, handling authentication and request formatting internally.

### Class Methods

##### _get_access_token()

This private method retrieves the Amadeus API access token using the API key and secret stored in environment variables. It's used internally by other methods to authenticate API requests.

##### search_flights()

Searches for flight offers using the Amadeus API. This method provides extensive flexibility in search parameters, allowing users to specify origin, destination, dates, number of travelers (including adults, children, and infants), travel class, non-stop preferences, currency, maximum price, and the number of results to return.

```python
AmadeusTools.search_flights(
    origin="NYC",
    destination="LON",
    departure_date="2023-07-01",
    return_date="2023-07-15",
    adults=2,
    children=1,
    infants=0,
    travel_class="ECONOMY",
    non_stop=False,
    currency="USD",
    max_price=1000,
    max_results=10
)
```

##### get_cheapest_date()

Finds the cheapest travel dates for a given route using the Flight Offers Search API. This method is particularly useful for flexible travel planning, allowing users to identify the most cost-effective dates for their journey.

```python
AmadeusTools.get_cheapest_date(
    origin="NYC",
    destination="PAR",
    departure_date="2023-08-01",
    return_date="2023-08-15",
    adults=2
)
```

##### get_flight_inspiration()

Retrieves flight inspiration using the Flight Inspiration Search API. This method is ideal for travelers who are open to various destinations, as it suggests travel options based on the origin city and optional price constraints. It's a powerful tool for discovering new travel opportunities and planning budget-friendly trips.

```python
AmadeusTools.get_flight_inspiration(
    origin="NYC",
    max_price=500,
    currency="USD"
)
```

Here's an example of how you might create a travel agent using these tools:

```python
travel_agent = Agent(
    role="Travel Planner",
    goal="Plan the most cost-effective and enjoyable trips for clients",
    attributes="Knowledgeable about global destinations, budget-conscious, detail-oriented",
    tools={AmadeusTools.search_flights, AmadeusTools.get_flight_inspiration},
    llm=OpenrouterModels.haiku
)
```

This travel agent can leverage the Amadeus tools to search for flights and get inspiration for travel destinations, making it a powerful assistant for travel planning tasks.

### Usage Notes

To use the AmadeusTools class, you must set the AMADEUS_API_KEY and AMADEUS_API_SECRET environment variables. These credentials are essential for authenticating with the Amadeus API and are securely managed by the class.

The class methods handle API authentication internally, abstracting away the complexity of token management. This allows developers to focus on making API calls and processing the returned data without worrying about the underlying authentication mechanism.

All methods in the AmadeusTools class return data in the form of Python dictionaries, making it easy to work with the results in your application. The structure of the returned data closely mirrors the JSON responses from the Amadeus API, ensuring that you have access to all the details provided by the API.

Error handling is built into these methods, with HTTP errors being caught and re-raised with additional context. This helps in debugging and handling potential issues that may arise during API interactions. These returned errors not only assist developers in debugging but also enable agents to self-correct, enhancing the robustness of the system.

