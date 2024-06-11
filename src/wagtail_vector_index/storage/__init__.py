from collections.abc import Mapping
from typing import TYPE_CHECKING

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

from .registry import registry  # noqa: F401

if TYPE_CHECKING:
    from wagtail_vector_index.storage.base import StorageProvider


class InvalidStorageProviderError(ImproperlyConfigured):
    pass


def get_storage_provider_config() -> Mapping:
    try:
        return settings.WAGTAIL_VECTOR_INDEX_STORAGE_PROVIDERS
    except AttributeError:
        return {
            "default": {
                "STORAGE_PROVIDER": "wagtail_vector_index.storage.numpy.NumpyStorageProvider",
            }
        }


def get_storage_provider(alias: str) -> "StorageProvider":
    provider_config = get_storage_provider_config()

    try:
        config = provider_config[alias]
    except KeyError as e:
        raise InvalidStorageProviderError(
            f"No storage provider with alias '{alias}': {e}"
        ) from e

    try:
        imported = import_string(config["STORAGE_PROVIDER"])
    except ImportError as e:
        raise InvalidStorageProviderError(
            f"Couldn't import storage provider {config['STORAGE_PROVIDER']}: {e}"
        ) from e

    params = config.copy()
    params.pop("STORAGE_PROVIDER")

    return imported(params)
