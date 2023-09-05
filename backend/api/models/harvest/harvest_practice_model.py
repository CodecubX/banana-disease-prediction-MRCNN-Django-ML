from django.db import models

from api.models.variety_model import Variety


class HarvestPractice(models.Model):
    """ Holds harvest practices """

    practice_name = models.CharField(max_length=100, null=False, blank=False)
    description = models.TextField()

    variety = models.ForeignKey(Variety, default=None, null=True, blank=True, on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.variety.variety} - {self.practice_name}'



