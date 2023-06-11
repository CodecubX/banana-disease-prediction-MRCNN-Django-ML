from django.db import models

from .disease_model import Disease


class Cure(models.Model):
    """ Holds Cure data """

    name = models.CharField(max_length=100, null=False, blank=False)
    description = models.TextField()

    img = models.ImageField(upload_to='cure/', max_length=500, null=True, blank=True)

    disease = models.ForeignKey(Disease, on_delete=models.CASCADE, null=False, blank=False)

    def __str__(self):
        return f'{self.disease.name} - {self.name}'
