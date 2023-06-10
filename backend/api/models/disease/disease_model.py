from django.db import models


class Disease(models.Model):
    """ Holds Disease data """

    name = models.CharField(max_length=100, null=False, blank=False, unique=True)
    description = models.TextField()

    img = models.ImageField(upload_to='disease/', null=True, blank=True)

    def __str__(self):
        return self.name
