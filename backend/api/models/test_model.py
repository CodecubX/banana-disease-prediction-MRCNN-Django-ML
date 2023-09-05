from django.db import models


class Test(models.Model):
    """ Holds variety model data """

    test = models.TextField()

    def __str__(self):
        return self.test