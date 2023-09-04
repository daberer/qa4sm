# Generated by Django 4.1 on 2023-07-13 12:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("validator", "0064_remove_dataset_filters_datasetversion_filters"),
    ]

    operations = [
        migrations.AlterField(
            model_name="user",
            name="space_limit",
            field=models.CharField(
                blank=True,
                choices=[
                    ("no_data", 1),
                    ("basic", 5000000000),
                    ("extended", 10000000000),
                    ("large", 200000000000),
                    ("unlimited", None),
                ],
                default="basic",
                max_length=25,
            ),
        ),
    ]