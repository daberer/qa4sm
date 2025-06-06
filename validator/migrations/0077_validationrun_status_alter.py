# Generated by Django 5.1.6 on 2025-02-27 06:05

import django.core.files.storage
from django.db import migrations, models
from validator.validation.util import determine_status


class Migration(migrations.Migration):

    def update_existing_statuses(apps, schema_editor):
        ValidationRun = apps.get_model('validator', 'ValidationRun')  # ✅ No direct import

        for instance in ValidationRun.objects.all():
            instance.status = determine_status(instance.progress, instance.end_time, instance.status)
            instance.save(update_fields=['status'])

    dependencies = [
        ('validator', '0076_auto_20241011_1155'),
    ]

    operations = [
        migrations.AddField(
            model_name='validationrun',
            name='status',
            field=models.CharField(choices=[('SCHEDULED', 'Scheduled'), ('RUNNING', 'Running'), ('DONE', 'Done'),
                                            ('CANCELLED', 'Cancelled'), ('ERROR', 'Error')], default='SCHEDULED',
                                   max_length=10),
        ),
        migrations.AddField(
            model_name='validationrun',
            name='is_removed',
            field=models.BooleanField(default=False),
        ),
        migrations.RunPython(update_existing_statuses)
    ]
