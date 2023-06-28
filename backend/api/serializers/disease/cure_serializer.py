from rest_framework import serializers

from api.models import Cure


class CureSerializer(serializers.ModelSerializer):
    """ Serializes the Cure model objects """

    name_display = serializers.CharField(source='name')

    class Meta:
        model = Cure
        fields = ['id', 'name_display', 'description', 'img', 'disease']


class CureSinhalaSerializer(serializers.ModelSerializer):
    """ Serializes the Cure model objects """

    name_display = serializers.CharField(source='name_sinhala')
    description = serializers.CharField(source='description_sinhala', allow_blank=True)

    class Meta:
        model = Cure
        fields = ['id', 'name_display', 'description', 'img', 'disease']