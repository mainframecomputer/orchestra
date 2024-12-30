# Stripe Tools

The StripeTools class provides a simplified interface to Stripe's API through the stripe_agent_toolkit package. It handles authentication and provides centralized access to common Stripe operations.

⚠️ **IMPORTANT DISCLAIMER** ⚠️

This toolkit provides direct access to Stripe API operations that can modify your Stripe account and process financial transactions. Please note:

- **Financial Risk**: Incorrect usage of these tools can result in unintended charges, refunds, or other financial consequences
- **No Warranty**: This toolkit is provided "AS IS" without any warranty. The authors and maintainers are not responsible for any financial losses or damages resulting from its use
- **Production Usage**: It is strongly recommended to:
  - Use test API keys during development
  - Implement additional safety checks and validation
  - Restrict access to write operations
  - Maintain audit logs of all operations

### Security Recommendations

1. **Access Control**: Implement role-based access control before allowing write operations
2. **Validation**: Add business-logic validation before executing operations
3. **Monitoring**: Log all operations and implement alerts for unusual activity
4. **Testing**: Always test with Stripe's test mode first


### Configuration

Before using Stripe operations, set up your environment variable:

```bash
export STRIPE_API_KEY=your_stripe_api_key
```

### Class Methods

##### check_balance()

Retrieve the current balance of your Stripe account.

```python
balance = StripeTools.check_balance()
```

##### list_customers(email: Optional[str] = None, limit: Optional[int] = None)

List customers from your Stripe account with optional filtering.

```python
# List all customers
customers = StripeTools.list_customers()

# Filter by email
customers = StripeTools.list_customers(email="user@example.com")

# Limit results
customers = StripeTools.list_customers(limit=10)
```

##### list_products(limit: Optional[int] = None)

List products from your Stripe catalog.

```python
products = StripeTools.list_products(limit=20)
```

##### create_customer(name: str, email: Optional[str] = None)

Create a new customer in Stripe.

```python
customer = StripeTools.create_customer(
    name="John Doe",
    email="john@example.com"
)
```

##### create_product(name: str, description: Optional[str] = None)

Create a new product in your Stripe catalog.

```python
product = StripeTools.create_product(
    name="Premium Plan",
    description="Access to all premium features"
)
```

##### create_price(product: str, currency: str, unit_amount: int)

Create a new price for a product.

```python
price = StripeTools.create_price(
    product="prod_xyz",
    currency="usd",
    unit_amount=1999  # $19.99
)
```

##### list_prices(product: Optional[str] = None, limit: Optional[int] = None)

List prices from your Stripe catalog.

```python
# List all prices
prices = StripeTools.list_prices()

# Filter by product
prices = StripeTools.list_prices(product="prod_xyz")
```

##### create_payment_link(price: str, quantity: int)

Create a payment link for a specific price.

```python
payment_link = StripeTools.create_payment_link(
    price="price_xyz",
    quantity=1
)
```

##### create_invoice(customer: str, days_until_due: int = 30)

Create a new invoice for a customer.

```python
invoice = StripeTools.create_invoice(
    customer="cus_xyz",
    days_until_due=14
)
```

##### create_invoice_item(customer: str, price: str, invoice: str)

Add an item to an invoice.

```python
item = StripeTools.create_invoice_item(
    customer="cus_xyz",
    price="price_xyz",
    invoice="inv_xyz"
)
```

##### finalize_invoice(invoice: str)

Finalize an invoice for sending.

```python
result = StripeTools.finalize_invoice("inv_xyz")
```

##### create_refund(payment_intent: str, amount: Optional[int] = None)

Create a refund for a payment.

```python
# Full refund
refund = StripeTools.create_refund("pi_xyz")

# Partial refund
refund = StripeTools.create_refund(
    payment_intent="pi_xyz",
    amount=1000  # $10.00
)
```

### Usage Notes

1. **Authentication**:
   - Requires `STRIPE_API_KEY` environment variable
   - Automatically handled through stripe_agent_toolkit

2. **Singleton Pattern**:
   - Uses a singleton pattern to maintain a single API instance
   - Automatically initializes on first use

3. **Currency Amounts**:
   - All amounts are in cents/smallest currency unit
   - Example: $10.00 = 1000 cents

4. **Response Format**:
   - All methods return JSON strings containing the response data
   - Parse the response using `json.loads()` if needed

5. **Error Handling**:
   - Errors from stripe_agent_toolkit are passed through
   - Check response for error details

6. **Rate Limiting**:
   - Respects Stripe's rate limits
   - Consider implementing retry logic for production use

7. **Best Practices**:
   - Always validate customer and product IDs
   - Use descriptive names for products and prices
   - Keep track of created resources

The StripeTools class simplifies Stripe operations by providing a centralized, authenticated interface to stripe_agent_toolkit's functionality. 