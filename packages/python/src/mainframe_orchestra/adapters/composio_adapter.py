from composio import LogLevel, ComposioToolSet, App
from typing import Optional, Set, Callable, Any, Tuple
from composio.client.collections import ActionModel
class ComposioAdapter:
    """Wrapper to convert Composio actions into callable functions for Orchestra"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the ComposioAdapter class

        Args:
            api_key: The API key to use for the ComposioAdapteret. If not provided, the API key will be read from the COMPOSIO_API_KEY environment variable.
        """
        self.composio_toolset = ComposioToolSet(logging_level=LogLevel.ERROR, lock=False, api_key=api_key)
        
    def connect(self, app: App) -> Tuple[str, str]:
        """
        Connect to an app and return the connection id and redirect url

        Args:
            app: The app to connect to

        Returns:
            A tuple containing redirect url, the connection id
        """
        integrations = self.composio_toolset.get_integrations(app)

        if len(integrations) == 0:
            raise ValueError(f"No integrations found for app {app}")

        integration = integrations[0]

        connection_request = self.composio_toolset.initiate_connection(
            integration_id=integration.id,
            entity_id="default",
        )

        return connection_request.connectedAccountId, connection_request.redirectUrl

    def get_tools(self, app: App, connection_id: Optional[str] = None) -> Set[Callable]:
        """
        Get the tools for an app

        Args:
            app: The app to get the tools for
            connection_id: The connection id to use. If not provided, the default connection id will be used.

        Returns:
            A set of tools
        """
        entity_id = "default" if not connection_id else None
        
        # Convert app names to App enum values if needed
        tool_schemas = self.composio_toolset.get_action_schemas(apps=[app])
        return [self._wrap(tool_schema, connection_id, entity_id) for tool_schema in tool_schemas]

    def _wrap(
            self, 
            tool_schema: ActionModel,
            connection_id: Optional[str] = None,
            entity_id: Optional[str] = None
        ) -> Callable:
        def wrapped_tool(**kwargs: Any) -> str:
            # Convert kwargs to a single string input
            output = self.composio_toolset.execute_action(
                tool_schema.name, 
                kwargs, 
                connected_account_id=connection_id, 
                entity_id=entity_id
            )
            return output

        # Build the docstring dynamically
        doc_parts = [
            f"- {tool_schema.name}:",
            f"    Description: {tool_schema.description}",
        ]
        params = tool_schema.parameters
        if params:
            doc_parts.append("    Arguments:")            
            for property_name, property in tool_schema.parameters.properties.items():
                argument_doc = "      - "
                if hasattr(property, "type"):
                    argument_doc += f"{property['type']} "
                else:
                    continue

                is_required = property_name in params.required if params.required else False
                argument_doc += f"{'' if is_required else ', optional'}"

                if hasattr(property, "description"):
                    argument_doc += f": {property['description']}"

                doc_parts.append(argument_doc)
        
        response = tool_schema.response
        if response:
            doc_parts.append("    Returns:")
            # return schema from a composio tool can get very complex, so we just return a generic type for now
            # It would be valuable to provide user a way to truncate the response for LLM consumption
            doc_parts.append(f"      Union[List[Dict], str]: tool call response, error message if failed")

        wrapped_tool.__name__ = tool_schema.name
        wrapped_tool.__doc__ = "\n".join(doc_parts)

        return wrapped_tool
