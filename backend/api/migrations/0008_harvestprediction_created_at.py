# Generated by Django 3.2.19 on 2023-06-10 06:59

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0007_alter_harvestprediction_predicted_harvest'),
    ]

    operations = [
        migrations.AddField(
            model_name='harvestprediction',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
    ]
