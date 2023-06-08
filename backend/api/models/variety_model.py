from django.db import models


class Variety(models.Model):
    """ Holds variety model data """

    variety = models.CharField(max_length=100, unique=True, null=False, blank=False)

    class Meta:
        verbose_name_plural = "Varieties"

    def __str__(self):
        return self.variety