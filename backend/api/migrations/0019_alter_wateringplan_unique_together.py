# Generated by Django 3.2.19 on 2023-07-11 15:10

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0018_rename_stunned_growth_diseasequestionnaireprediction_stunted_growth'),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='wateringplan',
            unique_together={('variety', 'stage', 'watering_plan')},
        ),
    ]