from django.db import models

from api.models import Variety


class AbstractBananaPlant(models.Model):
    """An abstract model that represents the common fields."""

    CHOICES = (
        ('vegetative', 'Vegetative'),
        ('pseudostem_formation', 'Pseudostem Formation'),
        ('shooting', 'Shooting'),
        ('inflorescence_initiation', 'Inflorescence Initiation'),
        ('flowering', 'Flowering'),
        ('fruit_development', 'Fruit Development'),
        ('harvest', 'Harvest'),
    )

    stage = models.CharField(choices=CHOICES, max_length=50, default='vegetative')
    variety = models.ForeignKey(Variety, on_delete=models.CASCADE, null=False, blank=False)

    class Meta:
        abstract = True
        unique_together = ('stage', 'variety')
