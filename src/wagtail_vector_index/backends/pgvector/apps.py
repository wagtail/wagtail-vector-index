from django.apps import AppConfig
from django.utils.translation import gettext_lazy as _


class PgvectorConfig(AppConfig):
    name = "wagtail_vector_index.backends.pgvector"
    verbose_name = _("pgvector")
    default_auto_field = "django.db.models.AutoField"

    # TODO: Check if we're using a Postgres database.
    #       Otherwise this app should not be allowed.
