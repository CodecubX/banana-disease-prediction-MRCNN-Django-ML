from rest_framework import serializers

from api.models import DiseasePrediction

from .disease_cure_serializer import DiseaseCureSerializer


class DiseasePredictionSerializer(serializers.ModelSerializer):
    """ Serializes the DiseasePrediction model objects and return selected fields """

    disease = DiseaseCureSerializer()
    original_img_url = serializers.ImageField(source='img', read_only=True)
    detected_img_url = serializers.ImageField(source='detected_img', read_only=True)

    class Meta:
        model = DiseasePrediction
        fields = ['id', 'user', 'probabilities', 'disease',  'original_img_url', 'detected_img_url', 'created_at']

