import asyncio
import logging

from channels.generic.http import AsyncHttpConsumer
from django import forms
from django.apps import apps
from django.http import QueryDict

logger = logging.Logger(__name__)


class WagtailVectorIndexQueryParamsForm(forms.Form):
    """Provides a form for validating query parameters."""

    query = forms.CharField(max_length=255, required=True)
    page_type = forms.CharField(max_length=255, required=True)


class WagtailVectorIndexSSEConsumer(AsyncHttpConsumer):
    """
    A Django Channels consumer for handling Server-Sent Events (SSE) related to WagtailVectorIndex queries.

    Methods:
        handle: The main entry point for processing HTTP requests, including SSE connections.
        process_prompt: Processes the incoming prompt and sends SSE updates.

    Note:
        This consumer expects the following query parameters in the URL:
        - 'query': The search query.
        - 'page_type': The type of Wagtail page to search.

        Example URL:
        "/chat-query-sse/?query=example&page_type=news.NewsPage"
    """

    async def handle(self, body):
        """
        Handles HTTP requests, sets up SSE headers, and processes prompts.
        """
        # Send SSE headers
        await self.send_headers(
            headers=[
                (b"Cache-Control", b"no-cache"),
                (b"Content-Type", b"text/event-stream"),
                (b"Transfer-Encoding", b"chunked"),
            ]
        )

        try:
            query_string = self.scope["query_string"].decode("utf-8")
            query_dict = QueryDict(query_string)

            # Validate query parameters
            form = WagtailVectorIndexQueryParamsForm(query_dict)
            if form.is_valid():
                query = form.cleaned_data["query"]
                page_type = form.cleaned_data["page_type"]

                # Get a model class by its name
                page_model = apps.get_model(page_type)
                vector_index = page_model.get_vector_index()

                try:
                    # Process and reply to prompt
                    await self.process_prompt(query, vector_index)
                except Exception:
                    logging.exception(
                        "Unexpected error in WagtailVectorIndexSSEConsumer"
                    )
                    payload = (
                        "data: Error processing request, Please try again later. \n\n"
                    )
                    await self.send_body(payload.encode("utf-8"), more_body=True)

        except (ValueError, UnicodeDecodeError, KeyError, LookupError, AttributeError):
            payload = "data: Error processing request. \n\n"
            await self.send_body(payload.encode("utf-8"), more_body=True)

        # Finish the response
        await self.send_body(b"")

    async def process_prompt(self, query, vector_index):
        """
        Processes the incoming prompt and sends SSE updates.

        Raises:
            asyncio.CancelledError: If the connection is cancelled or disconnected.
        """
        try:
            stream_response, _sources = await vector_index.aquery(query)
            for chunk in stream_response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content.replace(
                        "\n", "<br/>"
                    )  # Support line breaks
                    payload = f"data: {content or ''}\n\n"
                    await self.send_body(payload.encode("utf-8"), more_body=True)
        except asyncio.CancelledError:
            # Handle disconnects if needed, can occur from a server restart.
            # Note: Django < 5 doesn't recognise client disconnects
            pass
