from django.db import models


class FertilizerPlan(models.Model):
    """ Holds fertilizer plan model data """

    fertilizer_type_choices = [
        ('Organic Fertilizer', 'Organic Fertilizer'),
        ('Inorganic Fertilizer', 'Inorganic Fertilizer'),
        ('Slow-Release Fertilizer', 'Slow-Release Fertilizer'),
        ('Liquid Fertilizer', 'Liquid Fertilizer'),
        ('Balanced Fertilizer', 'Balanced Fertilizer'),
        ('Controlled-Release Fertilizer', 'Controlled-Release Fertilizer')
    ]

    fertilizer_type = models.CharField(choices=fertilizer_type_choices, max_length=200, null=False, blank=False, unique=True)
    description = models.TextField()

    def __str__(self):
        return self.get_fertilizer_type_display()
