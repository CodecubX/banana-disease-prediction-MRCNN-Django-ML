from django.db import models


class TestModel(models.Model):
    """ Holds variety model data """

    test = models.TextField()

