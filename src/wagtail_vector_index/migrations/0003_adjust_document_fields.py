# Generated by Django 5.0.1 on 2024-09-06 10:26

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("pgvector", "0003_alter_pgvectorembedding_embedding"),
        (
            "wagtail_vector_index",
            "0002_rename_embedding_model",
        ),
    ]

    operations = [
        migrations.RemoveField(
            model_name="document",
            name="base_content_type",
        ),
        migrations.RemoveField(
            model_name="document",
            name="content_type",
        ),
        migrations.RemoveField(
            model_name="document",
            name="object_id",
        ),
        migrations.AddField(
            model_name="document",
            name="object_key",
            field=models.CharField(max_length=255),
        ),
        migrations.AddField(
            model_name="document",
            name="object_keys",
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name="document",
            name="metadata",
            field=models.JSONField(default=dict),
        ),
    ]