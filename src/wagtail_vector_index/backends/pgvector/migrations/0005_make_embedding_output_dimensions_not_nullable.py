# Generated by Django 4.2.7 on 2023-12-05 11:38

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("pgvector", "0004_populate_embedding_output_dimensions"),
    ]

    operations = [
        migrations.AlterField(
            model_name="pgvectorembedding",
            name="embedding_output_dimensions",
            field=models.PositiveIntegerField(db_index=True),
        ),
    ]