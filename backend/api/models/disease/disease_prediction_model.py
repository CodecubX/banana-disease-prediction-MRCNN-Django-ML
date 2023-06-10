from django.db import models

from api.models import User


class DiseasePrediction(models.Model):
    """ Holds Disease predictions data """

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=False, blank=False)