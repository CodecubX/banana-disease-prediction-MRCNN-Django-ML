from django.db import models

from .variety_model import VarietyModel


class HarvestPracticesModel(models.Model):
    """ Holds harvest  practices """

    variety = models.ForeignKey(VarietyModel, on_delete=models.DO_NOTHING, null=False, blank=False)
    post_harvest_practices = models.TextField(null=False, blank=False)

    def __str__(self):
        return self.variety
