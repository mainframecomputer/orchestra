from composio import LogLevel, ComposioToolSet, App
from typing import Set, Callable, List, Any
from composio.client.collections import ActionModel
class ComposioTools:
    """Wrapper to convert Composio actions into callable functions for Orchestra"""

    def __init__(self):
        self.composio_toolset = ComposioToolSet(logging_level=LogLevel.ERROR, lock=False, api_key="kiycklo2ggdv24fi1fze")
        
    def get_tools(self, apps: List[str]) -> Set[Callable]:
        # Convert app names to App enum values if needed
        app_enums = []
        for app in apps:
            try:
                app_enum = getattr(App, app.upper())
                app_enums.append(app_enum)
            except AttributeError:
                print(f"Warning: App '{app}' not found in App enum, using string value")
                app_enums.append(app)

        tool_schemas = self.composio_toolset.get_action_schemas(apps=app_enums)
        return [ComposioTools._wrap(self.composio_toolset, tool_schema) for tool_schema in tool_schemas]

    @staticmethod
    def _wrap(toolset: ComposioToolSet, tool_schema: ActionModel):
        # Now proceed with the implementation
        def wrapped_tool(**kwargs: Any) -> str:
            # Convert kwargs to a single string input
            output = toolset.execute_action(tool_schema.name, kwargs)
            # output = langchain_tool.run(kwargs)
            return output

        # Build the docstring dynamically
        doc_parts = [
            f"- {tool_schema.name}:",
            f"    Description: {tool_schema.description}",
        ]

        if tool_schema.parameters:
            doc_parts.append("    Arguments:")
            print(f"Properties type: {type(tool_schema.parameters.properties)}")
            for property_name, property in tool_schema.parameters.properties.items():
                is_required = property_name in tool_schema.parameters.required
                doc_parts.append(f"      - {property_name} ({property['type']}{'' if is_required else ', optional'}): {property['description']}")
            
        wrapped_tool.__name__ = tool_schema.name
        wrapped_tool.__doc__ = "\n".join(doc_parts)

        return wrapped_tool
