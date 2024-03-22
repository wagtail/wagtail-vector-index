from django.apps import AppConfig


class WagtailVectorIndexAppConfig(AppConfig):
    label = "wagtail_vector_index"
    name = "wagtail_vector_index"
    verbose_name = "Wagtail Vector Index"
    default_auto_field = "django.db.models.AutoField"

    def ready(self):
        from wagtail_vector_index.index.models import register_indexed_models

        register_indexed_models()
