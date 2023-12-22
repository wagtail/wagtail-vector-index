from django.apps import AppConfig
from django.conf import settings


class WagtailVectorIndexAppConfig(AppConfig):
    label = "wagtail_vector_index"
    name = "wagtail_vector_index"
    verbose_name = "Wagtail Vector Index"

    def ready(self):
        update_on_publish = getattr(
            settings, "WAGTAIL_VECTOR_INDEX_UPDATE_ON_PUBLISH", False
        )
        if update_on_publish:
            # Register singles update indexes on publish
            from wagtail_vector_index import signals

            signals.register_signal_handlers()
