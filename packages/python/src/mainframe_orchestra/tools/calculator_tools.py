# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

from datetime import timedelta, datetime
from ..utils.braintrust_utils import traced

class CalculatorTools:
    @traced(type="tool")
    @staticmethod
    def basic_math(operation: str, args: list) -> float:
        """
        Perform basic and advanced math operations on multiple numbers.

        Args:
            operation (str): One of 'add', 'subtract', 'multiply', 'divide', 'exponent', 'root', 'modulo', or 'factorial'.
            args (list): List of numbers to perform the operation on.

        Returns:
            float: Result of the operation.

        Raises:
            ValueError: If an invalid operation is provided, if dividing by zero, if fewer than required numbers are provided, or for invalid inputs.

        Note:
            This method does not take in letters or words. It only takes in numbers.
        """
        if len(args) < 1:
            raise ValueError("At least one number is required for the operation.")

        # Convert all args to float, except for factorial which requires int
        if operation != 'factorial':
            args = [float(arg) for arg in args]

        result = args[0]

        if operation in ['add', 'subtract', 'multiply', 'divide']:
            if len(args) < 2:
                raise ValueError("At least two numbers are required for this operation.")

            if operation == 'add':
                for num in args[1:]:
                    result += num
            elif operation == 'subtract':
                for num in args[1:]:
                    result -= num
            elif operation == 'multiply':
                for num in args[1:]:
                    result *= num
            elif operation == 'divide':
                for num in args[1:]:
                    if num == 0:
                        raise ValueError("Cannot divide by zero")
                    result /= num
        elif operation == 'exponent':
            if len(args) != 2:
                raise ValueError("Exponent operation requires exactly two numbers.")
            result = args[0] ** args[1]
        elif operation == 'root':
            if len(args) != 2:
                raise ValueError("Root operation requires exactly two numbers.")
            if args[1] == 0:
                raise ValueError("Cannot calculate 0th root")
            result = args[0] ** (1 / args[1])
        elif operation == 'modulo':
            if len(args) != 2:
                raise ValueError("Modulo operation requires exactly two numbers.")
            if args[1] == 0:
                raise ValueError("Cannot perform modulo with zero")
            result = args[0] % args[1]
        elif operation == 'factorial':
            if len(args) != 1 or args[0] < 0 or not isinstance(args[0], int):
                raise ValueError("Factorial operation requires exactly one non-negative integer.")
            result = 1
            for i in range(1, args[0] + 1):
                result *= i
        else:
            raise ValueError("Invalid operation. Choose 'add', 'subtract', 'multiply', 'divide', 'exponent', 'root', 'modulo', or 'factorial'.")

        # Convert the result to a string before returning
        return str(result)

    @traced(type="tool")
    @staticmethod
    def get_current_time() -> str:
        """
        Get the current UTC time.

        Returns:
            str: The current UTC time in the format 'YYYY-MM-DD HH:MM:SS'.
        """
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    @traced(type="tool")
    @staticmethod
    def add_days(date_str: str, days: int) -> str:
        """
        Add a number of days to a given date.

        Args:
            date_str (str): The starting date in 'YYYY-MM-DD' format.
            days (int): The number of days to add (can be negative).

        Returns:
            str: The resulting date in 'YYYY-MM-DD' format.
        """
        date = datetime.strptime(date_str, "%Y-%m-%d")
        new_date = date + timedelta(days=days)
        return new_date.strftime("%Y-%m-%d")

    @traced(type="tool")
    @staticmethod
    def days_between(date1_str: str, date2_str: str) -> int:
        """
        Calculate the number of days between two dates.

        Args:
            date1_str (str): The first date in 'YYYY-MM-DD' format.
            date2_str (str): The second date in 'YYYY-MM-DD' format.

        Returns:
            int: The number of days between the two dates.
        """
        date1 = datetime.strptime(date1_str, "%Y-%m-%d")
        date2 = datetime.strptime(date2_str, "%Y-%m-%d")
        return abs((date2 - date1).days)

    @traced(type="tool")
    @staticmethod
    def format_date(date_str: str, input_format: str, output_format: str) -> str:
        """
        Convert a date string from one format to another.

        Args:
            date_str (str): The date string to format.
            input_format (str): The current format of the date string.
            output_format (str): The desired output format.

        Returns:
            str: The formatted date string.

        Example:
            format_date("2023-05-15", "%Y-%m-%d", "%B %d, %Y") -> "May 15, 2023"
        """
        date_obj = datetime.strptime(date_str, input_format)
        return date_obj.strftime(output_format)
