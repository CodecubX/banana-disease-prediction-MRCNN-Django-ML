from rest_framework import serializers

from api.models import Disease


class DiseaseSerializer(serializers.ModelSerializer):
    """ Serializes the Disease model objects """

    class Meta:
        model = Disease
        fields = '__all__'

