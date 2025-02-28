from fastmcp import FastMCP

# Simple FastMCP server
mcp = FastMCP("Calculator")

# Define tools
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers

    Args:
        a: The first number to add
        b: The second number to add

    Returns:
        The sum of the two numbers
    """
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers

    Args:
        a: The first number to multiply
        b: The second number to multiply

    Returns:
        The product of the two numbers
    """
    return a * b

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting

    Args:
        name: The name of the person to greet

    Returns:
        A personalized greeting
    """
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()