from django.db import models


class VarietyModel(models.Model):
    """ Holds variety model data """

    variety = models.CharField(max_length=100, null=False, blank=False)

    def __str__(self):
        return self.variety