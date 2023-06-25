from rest_framework import serializers

from api.models import Cure


class CureSerializer(serializers.ModelSerializer):
    """ Serializes the Cure model objects """

    class Meta:
        model = Cure
        fields = ['id', 'name', 'description', 'img', 'disease']


class CureSinhalaSerializer(serializers.ModelSerializer):
    """ Serializes the Cure model objects """

    name = serializers.CharField(source='name_sinhala')
    description = serializers.CharField(source='description_sinhala', allow_blank=True)

    class Meta:
        model = Cure
        fields = ['id', 'name', 'description', 'img', 'disease']