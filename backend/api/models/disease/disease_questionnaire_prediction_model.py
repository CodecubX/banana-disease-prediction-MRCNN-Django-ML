from django.db import models

from api.models import User
from .disease_model import Disease


class DiseaseQuestionnairePrediction(models.Model):
    """ Holds Disease Questionnaire predictions data """

    leaf_color = models.CharField(max_length=100)
    leaf_spots = models.CharField(max_length=100)
    leaf_wilting = models.CharField(max_length=100)
    leaf_curling = models.CharField(max_length=100)
    stunted_growth = models.CharField(max_length=100)
    stem_color = models.CharField(max_length=100)
    root_rot = models.CharField(max_length=100)
    abnormal_fruiting = models.CharField(max_length=100)
    presence_of_pests = models.CharField(max_length=100)

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=False, blank=False)
    # fields for prediction fields
    disease = models.ForeignKey(Disease, on_delete=models.DO_NOTHING, null=False, blank=False)
    probabilities = models.JSONField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.disease.name} - {self.user}'
