from django.db import models

from api.models import Variety, User


class HarvestPrediction(models.Model):
    """ Holds harvest prediction data """

    predicted_harvest = models.FloatField()
    agro_climatic_region = models.CharField(max_length=100)
    plant_density = models.IntegerField()
    spacing_between_plants = models.FloatField()
    pesticides_used = models.CharField(max_length=3)
    plant_generation = models.CharField(max_length=10)
    fertilizer_type = models.CharField(max_length=100)
    soil_pH = models.FloatField()
    amount_of_sunlight_received = models.CharField(max_length=100)
    watering_schedule = models.CharField(max_length=100)
    number_of_leaves = models.IntegerField()
    height = models.FloatField()

    variety = models.ForeignKey(Variety, on_delete=models.DO_NOTHING, null=False, blank=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=False, blank=False)

    def __str__(self):
        return f'{self.variety}: {self.harvest}'
