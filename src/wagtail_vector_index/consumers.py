import asyncio
import logging

from django.apps import apps
from django.core.exceptions import PermissionDenied
from channels.generic.http import AsyncHttpConsumer

logger = logging.Logger(__name__)


class WagtailVectorIndexSSEConsumer(AsyncHttpConsumer):
    """
    A Django Channels consumer for handling Server-Sent Events (SSE) related to WagtailVectorIndex queries.

    Attributes:
        page_instance (Model): The Wagtail page model instance for which the vector index is queried.
        page_model_name (str): The name of the Wagtail page model for which the vector index is queried.
        vector_index (VectorIndex): The vector index associated with the Wagtail page model.

    Methods:
        handle: The main entry point for processing HTTP requests, including SSE connections.
        process_prompt: Processes the incoming prompt and sends SSE updates.
        check_permissions: Checks user authentication and raises PermissionDenied if not authenticated.
        ratelimit_request: Placeholder for implementing rate-limiting logic.

    Usage:
    - Inherit from this class and set either 'page_instance' or 'page_model_name' attribute.
    - Implement custom logic within 'process_prompt' to handle vector index queries and SSE updates.
    - Optionally, override 'check_permissions' and 'ratelimit_request' for additional safety checks.

    Example:
    ```python
    class CustomSSEConsumer(WagtailVectorIndexSSEConsumer):
        page_instance = YourWagtailPageModel

        async def process_prompt(self, query):
            # Your custom logic to handle the query and send SSE updates
            pass
    ```
    """
    page_instance = None
    page_model_name = None

    def __init__(self, *args, **kwargs):
        """
        Initializes the consumer and checks the required attributes.

        Raises:
            ValueError: If neither 'page_instance' nor 'page_model_name' is set.
            ValueError: If the specified 'page_model_name' is not found.
            ValueError: If the specified page model does not inherit the ModelVectorIndex mixin.
        """
        super().__init__(*args, **kwargs)

        # Check if either page_model_name or page_instance is set
        if self.page_model_name is None and self.page_instance is None:
            raise ValueError('You must set either the page_model_name or page_instance attribute')

        if not self.page_instance:
            try:
                self.page_instance = apps.get_model(self.page_model_name)
            except LookupError:
                raise ValueError(f'Model {self.page_model_name} not found')

        # Check if the page model has the required method (ModelVectorIndex mixin)
        if not hasattr(self.page_instance, 'get_vector_index') or not callable(self.page_instance.get_vector_index):
            raise ValueError('Your page_model must inherit the ModelVectorIndex mixin')

        self.vector_index = self.page_instance.get_vector_index()

    async def handle(self, body):
        """
        Handles HTTP requests, sets up SSE headers, and processes prompts.

        Raises:
            PermissionDenied: If the user is not authenticated.
        """
        # Send SSE headers
        await self.send_headers(headers=[
            (b"Cache-Control", b"no-cache"),
            (b"Content-Type", b"text/event-stream"),
            (b"Transfer-Encoding", b"chunked"),
        ])

        try:
            try:
                user = self.scope['user']
            except KeyError as e:
                raise ValueError('User not found in scope, make sure AuthMiddlewareStack is applied correctly') from e
            
            await self.check_permissions(user)  # Check permissions
            await self.ratelimit_request()  # Apply rate limiting

            # Process and reply to prompt
            query = self.scope["query_string"].decode("utf-8")
            await self.process_prompt(query)

        except Exception as e:
            logging.exception("Unexpected error in WagtailVectorIndexSSEConsumer")
            payload = "data: Error processing request, Please try again later. \n\n"
            await self.send_body(payload.encode("utf-8"), more_body=True)

        # Finish the response
        await self.send_body(b"")

    async def process_prompt(self, query):
        """
        Processes the incoming prompt and sends SSE updates.

        Raises:
            asyncio.CancelledError: If the connection is cancelled or disconnected.
        """
        try:
            stream_response, sources = await self.vector_index.query_async(query)
            # TODO send or stream sources as characters to the client as well?
            for chunk in stream_response:
                if chunk.choices[0].delta.content is not None:
                    # TODO Remove after testing
                    # print(chunk.choices[0].delta.content, end="") # Uncomment to view response in terminal
                    # await asyncio.sleep(0.1) # Uncomment to test a more delayed response
                    content = chunk.choices[0].delta.content.replace('\n', '<br/>') # Support line breaks
                    payload = f"data: {content or ''}\n\n"
                    await self.send_body(payload.encode("utf-8"), more_body=True)
        except asyncio.CancelledError:
            # Handle disconnects if needed, can occur from a server restart.
            # Note: Django < 5 doesn't recognise client disconnects
            pass

    async def check_permissions(self, user):
        """
        Checks user authentication and raises PermissionDenied if not authenticated.

        Args:
            user: The authenticated user.

        Raises:
            PermissionDenied: If the user is not authenticated.
        """
        if not user.is_authenticated:
            # TODO log a 403, no way to send one via SSE, may need custom middleware?
            raise PermissionDenied("Permission denided")

    async def ratelimit_request(self):
        """
        Placeholder for implementing rate-limiting logic.

        Implement your custom rate-limiting logic within this method.
        """
        pass
