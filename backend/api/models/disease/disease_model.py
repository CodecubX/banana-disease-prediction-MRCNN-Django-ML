from django.db import models


class Disease(models.Model):
    """ Holds Disease data """

    # need to add the class labels the mrcnn model trained for (except bg class)
    name_choices = [
        ('Healthy', 'Healthy'),
        ('Yellow Sigatoka', 'Yellow Sigatoka'),
        ('Black Sigatoka', 'Black Sigatoka'),
        ('Panama', 'Panama'),
        ('Yellow Leaf', 'Yellow Leaf'),
        ('Anthracnose', 'Anthracnose'),
        ('Banana Bunchy Top', 'Banana Bunchy Top'),
        ('Fusarium Wilt', 'Fusarium Wilt'),
        ('Nutrient Deficiency', 'Nutrient Deficiency'),
        ('Nitrogen Deficiency', 'Nitrogen Deficiency'),
        ('Banana Mosaic', 'Banana Mosaic'),
        ('Magnesium Deficiency', 'Magnesium Deficiency'),
        ('Bacterial Wilt', 'Bacterial Wilt'),
        ('Yellow Sigatoka', 'Yellow Sigatoka'),
        ('Zinc Deficiency', 'Zinc Deficiency'),
        ('Phosphorus Deficiency', 'Phosphorus Deficiency'),
        ('Banana Streak Virus', 'Banana Streak Virus'),
        ('Potassium Deficiency', 'Potassium Deficiency'),
        ('Banana Fusarium Wilt', 'Banana Fusarium Wilt'),
        ('Iron Deficiency', 'Iron Deficiency'),
    ]

    name = models.CharField(choices=name_choices, max_length=200, null=False, blank=False, unique=True)
    description = models.TextField()
    description_sinhala = models.TextField()
    symptom_description = models.TextField()
    symptom_description_sinhala = models.TextField()

    img = models.ImageField(upload_to='disease/', max_length=500, null=True, blank=True)

    def __str__(self):
        return self.get_name_display()
