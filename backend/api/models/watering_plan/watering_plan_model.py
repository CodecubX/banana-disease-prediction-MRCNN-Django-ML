from django.db import models


class WateringPlan(models.Model):
    """ Holds water plan model data """

    watering_plan_choices = [
        ('every day', 'Every Day'),
        ('once every 2-3 days', 'Once Every 2-3 Days'),
        ('once every 3-4 days', 'Once Every 3-4 Days')
    ]

    watering_plan = models.CharField(choices=watering_plan_choices, max_length=200, null=False, blank=False, unique=True)
    stage = models.CharField(max_length=100)
    description = models.TextField()

    def __str__(self):
        return self.watering_plan
