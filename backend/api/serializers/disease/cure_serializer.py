from rest_framework import serializers

from api.models import Cure


class CureSerializer(serializers.ModelSerializer):
    """ Serializes the Cure model objects """

    class Meta:
        model = Cure
        fields = '__all__'

