# File Tools

The FileTools class provides a set of methods for performing various file-related operations, including saving code to files, generating directory trees, reading file contents, and searching within files of different formats (CSV, JSON, XML, YAML).

### Class Methods

##### save_code_to_file()

Saves the given code to a file at the specified path. It creates the necessary directories if they don't exist and handles potential errors that may occur during the file saving process.

```python
FileTools.save_code_to_file(
    code="print('Hello, World!')",
    file_path="path/to/file.py"
)
```

##### generate_directory_tree()

Recursively generates a file structure dictionary for the given base path. It traverses the directory tree, ignoring specified files and directories, and returns a nested dictionary representing the file structure. Each directory is represented by a dict with 'name', 'type', and 'children' keys, and each file is represented by a dict with 'name', 'type', and 'contents' keys.

```python
file_structure = FileTools.generate_directory_tree(
    base_path="path/to/directory",
    additional_ignore=[".git", "temp"]
)
```

##### read_file_contents()

Retrieves the contents of a file at the specified path. It attempts to read the file using UTF-8 encoding and falls back to ISO-8859-1 encoding if necessary. It returns the file contents as a string if successfully read, or None if an error occurs.

```python
file_contents = FileTools.read_file_contents("path/to/file.txt")
```

##### read_csv()

Reads a CSV file and returns its contents as a list of dictionaries, where each dictionary represents a row in the CSV. It uses the csv module to parse the CSV file and handles potential errors that may occur during the file reading process.

```python
csv_data = FileTools.read_csv("path/to/file.csv")
```

##### read_json()

Reads a JSON file and returns its contents as a dictionary or a list, depending on the structure of the JSON data. It uses the json module to parse the JSON file and handles potential errors that may occur during the file reading process.

```python
json_data = FileTools.read_json("path/to/file.json")
```

##### read_xml()

Reads an XML file and returns its contents as an ElementTree object. It uses the xml.etree.ElementTree module to parse the XML file and handles potential errors that may occur during the file reading process.

```python
xml_data = FileTools.read_xml("path/to/file.xml")
```

##### read_yaml()

Reads a YAML file and returns its contents as a dictionary or a list, depending on the structure of the YAML data. It uses the yaml module to parse the YAML file and handles potential errors that may occur during the file reading process.

```python
yaml_data = FileTools.read_yaml("path/to/file.yaml")
```

##### search_csv()

Searches for a specific value in a CSV file and returns matching rows as a list of dictionaries. It uses the pandas library to read the CSV file and perform the search based on the specified column and value.

```python
matching_rows = FileTools.search_csv(
    file_path="path/to/file.csv",
    search_column="name",
    search_value="John"
)
```

##### search_json()

Searches for a specific key-value pair in a JSON structure and returns matching items as a list. It recursively traverses the JSON data and appends matching items to the results list.

```python
matching_items = FileTools.search_json(
    data=json_data,
    search_key="age",
    search_value=30
)
```

##### search_xml()

Searches for specific elements in an XML structure based on tag name, attribute name, and attribute value. It uses the xml.etree.ElementTree module to traverse the XML tree and find matching elements.

```python
matching_elements = FileTools.search_xml(
    root=xml_data,
    tag="book",
    attribute="genre",
    value="fiction"
)
```

##### search_yaml()

Searches for a specific key-value pair in a YAML structure and returns matching items as a list. It reuses the search_json() method since YAML is parsed into Python data structures.

```python
matching_items = FileTools.search_yaml(
    data=yaml_data,
    search_key="name",
    search_value="Alice"
)
```

### Usage Notes

The FileTools class provides a convenient way to perform various file-related operations in Python. It offers methods for saving code to files, generating directory trees, reading file contents, and searching within files of different formats.

When using the save_code_to_file() method, ensure that you have the necessary write permissions for the specified file path. The method will create the necessary directories if they don't exist.

The generate_directory_tree() method allows you to generate a nested dictionary representation of a directory structure. You can specify additional files or directories to ignore using the additional_ignore parameter.

The read_file_contents() method attempts to read the file using UTF-8 encoding and falls back to ISO-8859-1 encoding if necessary. It returns the file contents as a string if successfully read, or None if an error occurs.

The read_csv(), read_json(), read_xml(), and read_yaml() methods provide convenient ways to read files of different formats and return their contents as Python data structures.

The search_csv(), search_json(), search_xml(), and search_yaml() methods allow you to search for specific values or elements within files of different formats. They return matching items as lists.

When using the file reading and searching methods, ensure that the specified file paths are correct and that you have the necessary read permissions for those files.

The FileTools class provides a set of static methods, which means you can directly call them using the class name without creating an instance of the class.

Remember to handle potential errors and exceptions that may occur during file operations, such as file not found errors or permission errors. The methods in the FileTools class raise appropriate exceptions in case of errors, which you can catch and handle accordingly in your code.

