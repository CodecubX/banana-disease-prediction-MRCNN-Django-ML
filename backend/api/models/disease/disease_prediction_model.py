from django.db import models

from api.models import User
from .disease_model import Disease


class DiseasePrediction(models.Model):
    """ Holds Disease predictions data """

    img = models.ImageField(upload_to='disease_prediction/original', null=True, blank=True)
    detected_img = models.ImageField(upload_to='disease_prediction/detected', null=True, blank=True)

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=False, blank=False)
    # fields for prediction fields
    disease = models.ForeignKey(Disease, on_delete=models.DO_NOTHING, null=False, blank=False)
    top_probabilities = models.JSONField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.disease.name} - {self.user}'