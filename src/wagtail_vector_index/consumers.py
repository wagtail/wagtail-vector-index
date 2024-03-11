import asyncio
import logging

from channels.generic.http import AsyncHttpConsumer
from django import forms
from django.core.exceptions import ValidationError
from django.http import QueryDict

from .base import VectorIndexableType

logger = logging.Logger(__name__)


class WagtailVectorIndexQueryParamsForm(forms.Form):
    """Provides a form for validating query parameters."""

    query = forms.CharField(max_length=255, required=True)
    index = forms.CharField(max_length=255, required=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from wagtail_vector_index.index import get_vector_indexes

        self.indexes = get_vector_indexes()

    def clean_index(self):
        index = self.cleaned_data["index"]
        if index not in self.indexes:
            raise forms.ValidationError("Invalid index. Please choose a valid index.")
        return index


class WagtailVectorIndexSSEConsumer(AsyncHttpConsumer):
    """
    A Django Channels consumer for handling Server-Sent Events (SSE) related to WagtailVectorIndex queries.

    Methods:
        handle: The main entry point for processing HTTP requests, including SSE connections.
        process_prompt: Processes the incoming prompt and sends SSE updates.

    Note:
        This consumer expects the following query parameters in the URL:
        - 'query': The search query.
        - 'index': The vector index to perform the query with.

        Example URL:
        "/chat-query-sse/?query=example&index=news.NewsPage"
    """

    async def handle(self, body: bytes) -> None:
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
            if not form.is_valid():
                # Ignore "TRY301 Abstract `raise` to an inner function"
                # So we can insure the event-stream is closed and no other code is executed
                raise ValidationError("Invalid query parameters.")  # noqa: TRY301
            query = form.cleaned_data["query"]
            index = form.cleaned_data["index"]

            vector_index = form.indexes.get(index)

            if vector_index:
                await self.process_prompt(query, vector_index)

        except ValidationError:
            await self.error_response()

        except Exception:
            logging.exception("Unexpected error in WagtailVectorIndexSSEConsumer")
            await self.error_response()

        # Finish the response
        await self.send_body(b"")

    async def error_response(self) -> None:
        payload = "data: Error processing request, Please try again later. \n\n"
        await self.send_body(payload.encode("utf-8"), more_body=True)

    async def process_prompt(
        self, query: str, vector_index: VectorIndexableType
    ) -> None:
        """
        Processes the incoming prompt and sends SSE updates.

        Raises:
            asyncio.CancelledError: If the connection is cancelled or disconnected.
        """
        try:
            results = await vector_index.aquery(query)
            for chunk in results.response:
                chunk = chunk.replace(
                    "\n", "<br/>"
                )  # Replace newlines with HTML line breaks to avoid issues with encoding.
                payload = f"data: {chunk}\n\n"  # Each message must be terminated using two newline characters.
                await self.send_body(payload.encode("utf-8"), more_body=True)
        except asyncio.CancelledError:
            # Handle disconnects if needed, can occur from a server restart.
            # Note: Django < 5 doesn't recognise client disconnects
            pass
