from rest_framework import serializers

from api.models.variety_model import Variety


class VarietySerializer(serializers.ModelSerializer):
    """ Serializes the variety model """

    class Meta:
        model = Variety
        fields = '__all__'
