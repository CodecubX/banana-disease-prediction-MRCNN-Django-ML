from django.db import models

from .watering_plan_model import WateringPlan
from api.models import User


class WateringPlanPrediction(models.Model):
    """ Holds water plan model data """

    pH = models.DecimalField(max_digits=3, decimal_places=1)
    organic_matter_content = models.CharField(max_length=20)
    soil_type = models.CharField(max_length=50)
    soil_moisture = models.CharField(max_length=10)
    avg_temperature = models.IntegerField()
    avg_rainfall = models.IntegerField()
    plant_height = models.IntegerField()
    leaf_color = models.CharField(max_length=20)
    stem_diameter = models.IntegerField()
    plant_density = models.IntegerField()
    soil_texture = models.CharField(max_length=20)
    soil_color = models.CharField(max_length=20)
    temperature = models.IntegerField()
    humidity = models.IntegerField()
    rainfall = models.IntegerField()
    water_source = models.CharField(max_length=20)
    irrigation_method = models.CharField(max_length=20)
    fertilizer_used_last_season = models.CharField(max_length=20)
    crop_rotation = models.CharField(max_length=5)
    pest_disease_infestation = models.CharField(max_length=5)
    slope = models.CharField(max_length=10)

    watering_plan = models.ForeignKey(WateringPlan, on_delete=models.DO_NOTHING, null=False, blank=False)
    top_probabilities = models.JSONField()
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=False, blank=False)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.watering_plan.watering_plan
