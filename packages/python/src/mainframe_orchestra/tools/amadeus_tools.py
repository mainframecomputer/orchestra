# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import os
from typing import Dict, Any, Optional, Tuple, Union
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
from ..utils.braintrust_utils import traced

class AmadeusTools:
    @staticmethod
    def _get_access_token():
        """Get Amadeus API access token."""
        load_dotenv()
        api_key = os.getenv("AMADEUS_API_KEY")
        api_secret = os.getenv("AMADEUS_API_SECRET")

        if not api_key or not api_secret:
            raise ValueError("AMADEUS_API_KEY and AMADEUS_API_SECRET must be set in .env file")

        token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": api_key,
            "client_secret": api_secret
        }

        response = requests.post(token_url, data=data)
        response.raise_for_status()
        return response.json()["access_token"]

    @traced(type="tool")
    @staticmethod
    def search_flights(
        origin: str,
        destination: str,
        departure_date: str,
        return_date: Optional[str] = None,
        adults: int = 1,
        children: int = 0,
        infants: int = 0,
        travel_class: Optional[str] = None,
        non_stop: bool = False,
        currency: str = "USD",
        max_price: Optional[int] = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search for flight offers using Amadeus API.

        Args:
            origin (str): Origin airport (e.g., "YYZ")
            destination (str): Destination airport (e.g., "CDG")
            departure_date (str): Departure date in YYYY-MM-DD format
            return_date (Optional[str]): Return date in YYYY-MM-DD format for round trips
            adults (int): Number of adult travelers
            children (int): Number of child travelers
            infants (int): Number of infant travelers
            travel_class (Optional[str]): Preferred travel class (ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST)
            non_stop (bool): If True, search for non-stop flights only
            currency (str): Currency code for pricing (default: USD)
            max_price (Optional[int]): Maximum price per traveler
            max_results (int): Maximum number of results to return (default: 10)

        Returns:
            Dict[str, Any]: Flight search results
        """
        access_token = AmadeusTools._get_access_token()

        url = "https://test.api.amadeus.com/v2/shopping/flight-offers"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        params = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": departure_date,
            "adults": adults,
            "children": children,
            "infants": infants,
            "currencyCode": currency,
            "max": max_results
        }

        if return_date:
            params["returnDate"] = return_date

        if travel_class:
            params["travelClass"] = travel_class

        if non_stop:
            params["nonStop"] = "true"

        if max_price:
            params["maxPrice"] = max_price

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_message = f"Error searching for flight offers: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_message += f"\nResponse status code: {e.response.status_code}"
                error_message += f"\nResponse content: {e.response.text}"
            return error_message

    @traced(type="tool")
    @staticmethod
    def get_cheapest_date(
        origin: str,
        destination: str,
        departure_date: Union[str, Tuple[str, str]],
        return_date: Optional[Union[str, Tuple[str, str]]] = None,
        adults: int = 1
    ) -> Dict[str, Any]:
        """
        Find the cheapest flight offer for a given route and date or date range using the Amadeus Flight Offers Search API.
        Max range of 7 days between start and end date.

        Args:
            origin (str): IATA code of the origin airport.
            destination (str): IATA code of the destination airport.
            departure_date (Union[str, Tuple[str, str]]): Departure date in YYYY-MM-DD format or a tuple of (start_date, end_date).
            return_date (Optional[Union[str, Tuple[str, str]]]): Return date in YYYY-MM-DD format or a tuple of (start_date, end_date) for round trips. Defaults to None.
            adults (int): Number of adult travelers. Defaults to 1.

        Returns:
            Dict[str, Any]: A dictionary containing the cheapest flight offer information.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
            ValueError: If the date range is more than 7 days.
        """
        access_token = AmadeusTools._get_access_token()

        url = "https://test.api.amadeus.com/v2/shopping/flight-offers"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        def date_range(start_date: str, end_date: str):
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            if (end - start).days > 7:
                return {"error": "Date range cannot exceed 7 days"}
            date = start
            while date <= end:
                yield date.strftime("%Y-%m-%d")
                date += timedelta(days=1)

        departure_dates = [departure_date] if isinstance(departure_date, str) else list(date_range(*departure_date))
        return_dates = [return_date] if return_date and isinstance(return_date, str) else (list(date_range(*return_date)) if return_date else [None])

        cheapest_offer = None
        cheapest_price = float('inf')

        for dep_date in departure_dates:
            for ret_date in return_dates:
                params = {
                    "originLocationCode": origin,
                    "destinationLocationCode": destination,
                    "departureDate": dep_date,
                    "adults": adults,
                    "max": 1,
                    "currencyCode": "USD"
                }
                if ret_date:
                    params["returnDate"] = ret_date

                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()

                data = response.json()
                if data.get('data'):
                    offer = data['data'][0]
                    price = float(offer['price']['total'])
                    if price < cheapest_price:
                        cheapest_price = price
                        cheapest_offer = offer

        if not cheapest_offer:
            return {"error": "No flights found for the given criteria"}

        result = {
            "price": cheapest_offer['price']['total'],
            "departureDate": cheapest_offer['itineraries'][0]['segments'][0]['departure']['at'],
            "airline": cheapest_offer['validatingAirlineCodes'][0],
            "details": cheapest_offer
        }

        if return_date:
            result["returnDate"] = cheapest_offer['itineraries'][-1]['segments'][0]['departure']['at']

        return result

    @traced(type="tool")
    @staticmethod
    def get_flight_inspiration(
        origin: str,
        max_price: Optional[int] = None,
        currency: str = "EUR"
    ) -> Dict[str, Any]:
        """
        Get flight inspiration using the Flight Inspiration Search API.

        This method uses the Amadeus Flight Inspiration Search API to find travel destinations
        based on the origin city and optional price constraints.

        Args:
            origin (str): IATA code of the origin city.
            max_price (Optional[int], optional): Maximum price for the flights. Defaults to None.
            currency (str, optional): Currency for the price. Defaults to "EUR".

        Returns:
            Dict[str, Any]: A dictionary containing flight inspiration results, including:
                - data: A list of dictionaries, each representing a destination with details such as:
                    - type: The type of the result (usually "flight-destination").
                    - origin: The IATA code of the origin city.
                    - destination: The IATA code of the destination city.
                    - departureDate: The suggested departure date.
                    - returnDate: The suggested return date.
                    - price: The price information for the trip.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
            ValueError: If the required environment variables are not set.

        Note:
            This method requires valid Amadeus API credentials to be set in the environment variables.
        """
        access_token = AmadeusTools._get_access_token()

        url = "https://test.api.amadeus.com/v1/shopping/flight-destinations"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        params = {
            "origin": origin,
            "currency": currency
        }

        if max_price:
            params["maxPrice"] = max_price

        response = requests.get(url, headers=headers, params=params)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response content: {response.text}")
            raise
        return response.json()

