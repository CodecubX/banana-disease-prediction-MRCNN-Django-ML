from rest_framework import serializers

from api.models import Disease

from .cure_serializer import CureSerializer


class DiseaseCureSerializer(serializers.ModelSerializer):
    """ Serializes the Disease model objects """

    cures = CureSerializer(many=True, read_only=True)  # Nested serialization for associated cures

    class Meta:
        model = Disease
        fields = ['name', 'description', 'img', 'cures']

