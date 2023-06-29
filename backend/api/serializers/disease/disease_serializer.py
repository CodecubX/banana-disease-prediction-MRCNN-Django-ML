from rest_framework import serializers

from api.models import Disease
from api.serializers import DynamicFieldsModelSerializer


class DiseaseSerializer(DynamicFieldsModelSerializer):
    """ Serializes the Disease model objects """

    name_display = serializers.CharField(source='get_name_display', read_only=True)

    class Meta:
        model = Disease
        fields = ['id', 'name', 'name_display', 'description', 'img', 'symptom_description']


class DiseaseSinhalaSerializer(DynamicFieldsModelSerializer):
    """ Serializes the Disease model objects """

    name_display = serializers.CharField(source='get_name_display', read_only=True)
    description = serializers.CharField(source='description_sinhala', allow_blank=True)
    symptom_description = serializers.CharField(source='symptom_description_sinhala', allow_blank=True)

    class Meta:
        model = Disease
        fields = ['id', 'name', 'name_display', 'description', 'img', 'symptom_description']

