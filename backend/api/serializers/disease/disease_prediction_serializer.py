from rest_framework import serializers

from api.models import DiseasePrediction

from .disease_serializer import DiseaseSerializer
from .cure_serializer import CureSerializer


class DiseasePredictionSerializer(serializers.ModelSerializer):
    """ Serializes the DiseasePrediction model objects and return selected fields """

    disease = DiseaseSerializer()
    cures = CureSerializer(source='disease.cure_set', many=True)

    class Meta:
        model = DiseasePrediction
        fields = ['id', 'user', 'top_probabilities', 'disease', 'cures', 'img', 'detected_img', 'created_at']

