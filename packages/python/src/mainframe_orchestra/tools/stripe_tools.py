# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

from typing import Optional
from dotenv import load_dotenv
import os
from stripe_agent_toolkit.api import StripeAPI
from stripe_agent_toolkit.configuration import Context

class StripeTools:
    _instance = None

    def __init__(self):
        load_dotenv()
        self.api = StripeAPI(
            secret_key=os.getenv('STRIPE_API_KEY'),
            context=Context()
        )

    @classmethod
    def get_api(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance.api

    @classmethod
    def check_balance(cls) -> str:
        """
        Retrieve the current balance of your Stripe account.
        
        Returns:
            str: JSON string containing available and pending balances in each currency
        """
        return cls.get_api().run("retrieve_balance")

    @classmethod
    def list_customers(cls, email: Optional[str] = None, limit: Optional[int] = None) -> str:
        """
        List customers from your Stripe account with optional filtering.

        Args:
            email (str, optional): Filter customers by email address
            limit (int, optional): Maximum number of customers to return (1-100)

        Returns:
            str: JSON string containing list of customer objects
        """
        return cls.get_api().run("list_customers", email=email, limit=limit)

    @classmethod
    def list_products(cls, limit: Optional[int] = None) -> str:
        """
        List products from your Stripe catalog.

        Args:
            limit (int, optional): Maximum number of products to return (1-100)

        Returns:
            str: JSON string containing list of product objects
        """
        return cls.get_api().run("list_products", limit=limit)

    @classmethod
    def create_customer(cls, name: str, email: Optional[str] = None) -> str:
        """
        Create a new customer in Stripe.

        Args:
            name (str): The customer's full name
            email (str, optional): The customer's email address

        Returns:
            str: JSON string containing the created customer object
        """
        return cls.get_api().run("create_customer", name=name, email=email)

    @classmethod
    def create_product(cls, name: str, description: Optional[str] = None) -> str:
        """
        Create a new product in your Stripe catalog.

        Args:
            name (str): Name of the product
            description (str, optional): Detailed description of the product

        Returns:
            str: JSON string containing the created product object
        """
        return cls.get_api().run("create_product", name=name, description=description)

    @classmethod
    def create_price(cls, product: str, currency: str, unit_amount: int) -> str:
        """
        Create a new price for a product in Stripe.

        Args:
            product (str): The ID of the product
            currency (str): Three-letter currency code (e.g., 'usd')
            unit_amount (int): Price amount in cents/smallest currency unit

        Returns:
            str: JSON string containing the created price object
        """
        return cls.get_api().run("create_price", product=product, currency=currency, unit_amount=unit_amount)

    @classmethod
    def list_prices(cls, product: Optional[str] = None, limit: Optional[int] = None) -> str:
        """
        List prices from your Stripe catalog.

        Args:
            product (str, optional): Filter prices by product ID
            limit (int, optional): Maximum number of prices to return (1-100)

        Returns:
            str: JSON string containing list of price objects
        """
        return cls.get_api().run("list_prices", product=product, limit=limit)

    @classmethod
    def create_payment_link(cls, price: str, quantity: int) -> str:
        """
        Create a payment link for a specific price.

        Args:
            price (str): The ID of the price
            quantity (int): The quantity of the product

        Returns:
            str: JSON string containing the payment link object with URL
        """
        return cls.get_api().run("create_payment_link", price=price, quantity=quantity)

    @classmethod
    def create_invoice(cls, customer: str, days_until_due: int = 30) -> str:
        """
        Create a new invoice for a customer.

        Args:
            customer (str): The ID of the customer
            days_until_due (int, optional): Number of days until invoice is due

        Returns:
            str: JSON string containing the created invoice object
        """
        return cls.get_api().run("create_invoice", customer=customer, days_until_due=days_until_due)

    @classmethod
    def create_invoice_item(cls, customer: str, price: str, invoice: str) -> str:
        """
        Add an item to an invoice.

        Args:
            customer (str): The ID of the customer
            price (str): The ID of the price
            invoice (str): The ID of the invoice

        Returns:
            str: JSON string containing the created invoice item object
        """
        return cls.get_api().run("create_invoice_item", customer=customer, price=price, invoice=invoice)

    @classmethod
    def finalize_invoice(cls, invoice: str) -> str:
        """
        Finalize an invoice for sending.

        Args:
            invoice (str): The ID of the invoice

        Returns:
            str: JSON string containing the finalized invoice object
        """
        return cls.get_api().run("finalize_invoice", invoice=invoice)

    @classmethod
    def create_refund(cls, payment_intent: str, amount: Optional[int] = None) -> str:
        """
        Create a refund for a payment.

        Args:
            payment_intent (str): The ID of the payment intent to refund
            amount (int, optional): Amount to refund in cents. If not provided, refunds entire payment.

        Returns:
            str: JSON string containing the created refund object
        """
        return cls.get_api().run("create_refund", payment_intent=payment_intent, amount=amount)
