# Generated by Django 5.1.6 on 2025-04-29 15:02

import django.core.files.storage
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('validator', '0079_validationrundeleted'),
    ]

    operations = [
        migrations.AddField(
            model_name='datasetversion',
            name='variables',
            field=models.ManyToManyField(related_name='variables', to='validator.datavariable'),
        ),
    ]
