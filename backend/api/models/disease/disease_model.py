from django.db import models


class Disease(models.Model):
    """ Holds Disease data """

    # need to add the class labels the mrcnn model trained for (except bg class)
    name_choices = [
        ('healthy', 'Healthy'),
        ('yellow_leaf', 'Yellow Leaf'),
        ('anthracnose', 'Anthracnose'),
    ]

    name = models.CharField(choices=name_choices, max_length=200, null=False, blank=False, unique=True)
    description = models.TextField()
    description_sinhala = models.TextField()
    symptom_description = models.TextField()
    symptom_description_sinhala = models.TextField()

    img = models.ImageField(upload_to='disease/', max_length=500, null=True, blank=True)

    def __str__(self):
        return self.get_name_display()
