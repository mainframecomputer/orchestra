# Copyright 2025 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

from typing import Any, Dict, List, Optional

from ..utils.braintrust_utils import traced


class LangchainTools:
    _module_lookup = None

    @classmethod
    def _get_module_lookup(cls):
        """Lazy load the langchain module lookup with proper error handling."""
        if cls._module_lookup is None:
            try:
                from langchain_community.tools import _module_lookup  # type: ignore

                cls._module_lookup = _module_lookup
            except ImportError as e:
                raise ImportError(
                    "langchain-community is not installed. "
                    "To use LangchainTools, install it with: "
                    "'pip install langchain-community'"
                ) from e
        return cls._module_lookup

    @classmethod
    def check_dependencies(cls) -> Dict[str, bool]:
        """
        Check if required dependencies are available.

        Returns:
            Dict[str, bool]: Dictionary with dependency names as keys and availability as values.
        """
        dependencies = {}

        try:
            cls._get_module_lookup()
            dependencies["langchain-community"] = True
        except ImportError:
            dependencies["langchain-community"] = False

        return dependencies

    @staticmethod
    def _wrap(langchain_tool):
        # Import optional dependencies inside the method
        import json
        from typing import Any

        from pydantic.v1 import BaseModel

        # Now proceed with the implementation
        def wrapped_tool(**kwargs: Any) -> str:
            tool_instance = langchain_tool()
            # Convert kwargs to a single string input
            tool_input = json.dumps(kwargs)
            return tool_instance.run(tool_input)

        tool_instance = langchain_tool()
        name = getattr(tool_instance, "name", langchain_tool.__name__)
        description = getattr(tool_instance, "description", "No description available")

        # Build the docstring dynamically
        doc_parts = [
            f"- {name}:",
            f"    Description: {description}",
        ]

        args_schema = getattr(langchain_tool, "args_schema", None) or getattr(
            tool_instance, "args_schema", None
        )
        if args_schema and issubclass(args_schema, BaseModel):
            doc_parts.append("    Arguments:")
            for field_name, field in args_schema.__fields__.items():
                field_desc = field.field_info.description or "No description"
                doc_parts.append(f"      - {field_name}: {field_desc}")

        wrapped_tool.__name__ = name
        wrapped_tool.__doc__ = "\n".join(doc_parts)
        return wrapped_tool

    @classmethod
    @traced(type="tool")
    def get_tool(cls, tool_name: str) -> Optional[Any]:
        """
        Get a wrapped Langchain tool.

        Args:
            tool_name (str): The name of the Langchain tool.

        Returns:
            Optional[Any]: The wrapped tool function, or None if dependencies are not available.
        """
        try:
            module_lookup = cls._get_module_lookup()
        except ImportError:
            print(
                "Error: langchain-community is not installed. Please install it using 'pip install langchain-community'."
            )
            return None

        if tool_name not in module_lookup:
            raise ValueError(f"Unknown Langchain tool: {tool_name}")

        module_path = module_lookup[tool_name]
        import importlib

        module = importlib.import_module(module_path)
        tool_class = getattr(module, tool_name)

        wrapped_tool = LangchainTools._wrap(tool_class)
        return wrapped_tool

    @classmethod
    def list_available_tools(cls) -> List[str]:
        """
        List all available Langchain tools.

        Returns:
            List[str]: A list of names of all available Langchain tools.

        Raises:
            ImportError: If langchain-community is not installed.

        Example:
            >>> tools = LangchainTools.list_available_tools()
            >>> "WikipediaQueryRun" in tools
            True
        """
        try:
            module_lookup = cls._get_module_lookup()
            return list(module_lookup.keys())
        except ImportError:
            print(
                "Error: langchain-community is not installed. Please install it using 'pip install langchain-community'."
            )
            return []

    @classmethod
    @traced(type="tool")
    def get_tool_info(cls, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve information about a specific Langchain tool.

        Args:
            tool_name (str): The name of the Langchain tool.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the tool's name, description, and module path.
            None: If required dependencies are not available.

        Raises:
            ValueError: If an unknown tool name is provided.

        Example:
            >>> info = LangchainTools.get_tool_info("WikipediaQueryRun")
            >>> "name" in info and "description" in info and "module_path" in info
            True
        """
        try:
            module_lookup = cls._get_module_lookup()
        except ImportError:
            print(
                "Error: langchain-community is not installed. Please install it using 'pip install langchain-community'."
            )
            return None

        if tool_name not in module_lookup:
            raise ValueError(f"Unknown Langchain tool: {tool_name}")

        module_path = module_lookup[tool_name]
        import importlib

        module = importlib.import_module(module_path)
        tool_class = getattr(module, tool_name)

        tool_instance = tool_class()
        name = getattr(tool_instance, "name", tool_class.__name__)
        description = getattr(tool_instance, "description", "No description available")

        return {"name": name, "description": description, "module_path": module_path}
