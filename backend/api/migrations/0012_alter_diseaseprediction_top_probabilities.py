# Generated by Django 3.2.19 on 2023-06-11 10:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0011_alter_disease_name'),
    ]

    operations = [
        migrations.AlterField(
            model_name='diseaseprediction',
            name='top_probabilities',
            field=models.JSONField(blank=True, null=True),
        ),
    ]