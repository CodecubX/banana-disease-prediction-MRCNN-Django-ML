from rest_framework import serializers

from api.models import Disease


class DiseaseSerializer(serializers.ModelSerializer):
    """ Serializes the Disease model objects """

    name_display = serializers.CharField(source='get_name_display', read_only=True)

    class Meta:
        model = Disease
        fields = '__all__'

