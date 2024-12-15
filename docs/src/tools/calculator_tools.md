# Calculator Tools

The CalculatorTools class provides a set of methods for performing basic and advanced math operations, date manipulation, and utility functions. It allows users to perform calculations, format dates, and retrieve the current time.

### Class Methods

##### basic_math()

Performs basic and advanced math operations on multiple numbers. It takes an operation string and a list of numbers as arguments and returns the result of the operation.

```python
CalculatorTools.basic_math(
    operation="add",
    args=[5, 3, 2]
)
```

##### get_current_time()

Retrieves the current UTC time in the format 'YYYY-MM-DD HH:MM:SS'.

```python
CalculatorTools.get_current_time()
```

##### add_days()

Adds a specified number of days to a given date and returns the resulting date in 'YYYY-MM-DD' format.

```python
CalculatorTools.add_days(
    date_str="2023-05-15",
    days=7
)
```

##### days_between()

Calculates the number of days between two dates provided in 'YYYY-MM-DD' format.

```python
CalculatorTools.days_between(
    date1_str="2023-05-01",
    date2_str="2023-05-15"
)
```

##### format_date()

Converts a date string from one format to another. It takes the date string, input format, and desired output format as arguments.

```python
CalculatorTools.format_date(
    date_str="2023-05-15",
    input_format="%Y-%m-%d",
    output_format="%B %d, %Y"
)
```

### Usage Notes

The basic_math() method supports various math operations, including 'add', 'subtract', 'multiply', 'divide', 'exponent', 'root', 'modulo', and 'factorial'. It requires at least one number for the operation, and some operations may require additional numbers.

The date-related methods (add_days(), days_between(), format_date()) expect dates to be provided in the 'YYYY-MM-DD' format by default. The format_date() method allows you to specify custom input and output formats.

The get_current_time() method returns the current UTC time as a string in the format 'YYYY-MM-DD HH:MM:SS'.

