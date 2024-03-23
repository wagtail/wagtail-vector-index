from collections.abc import Mapping
from typing import TYPE_CHECKING

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

if TYPE_CHECKING:
    from wagtail_vector_index.backends.base import Backend


class InvalidVectorBackendError(ImproperlyConfigured):
    pass


def get_vector_backend_config() -> Mapping:
    try:
        return settings.WAGTAIL_VECTOR_INDEX_VECTOR_BACKENDS
    except AttributeError:
        return {
            "default": {
                "BACKEND": "wagtail_vector_index.backends.numpy.NumpyBackend",
            }
        }


def get_vector_backend(alias: str) -> "Backend":
    backend_config = get_vector_backend_config()

    try:
        config = backend_config[alias]
    except KeyError as e:
        raise InvalidVectorBackendError(
            f"No vector backend with alias '{alias}': {e}"
        ) from e

    try:
        imported = import_string(config["BACKEND"])
    except ImportError as e:
        raise InvalidVectorBackendError(
            f"Couldn't import backend {config['BACKEND']}: {e}"
        ) from e

    params = config.copy()
    params.pop("BACKEND")

    return imported(params)
