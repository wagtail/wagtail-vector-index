from django.apps import apps
from wagtail.signals import page_published


def run_command_on_publish(sender, **kwargs):
    # Check page is live as not to leak draft content
    if kwargs["instance"].live:
        if hasattr(kwargs["instance"], "get_vector_index"):
            index = kwargs["instance"].get_vector_index()
            index.rebuild_index()


def register_signal_handlers():
    from wagtail_vector_index.models import VectorIndexedMixin

    indexes = [
        model
        for model in apps.get_models()
        if issubclass(model, VectorIndexedMixin) and not model._meta.abstract
    ]

    for index in indexes:
        page_published.connect(run_command_on_publish, sender=index)
