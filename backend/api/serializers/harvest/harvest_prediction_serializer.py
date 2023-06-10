from rest_framework import serializers

from api.models import HarvestPrediction

from .harvest_practice_serializer import HarvestPracticeSerializer


class HarvestPredictionSerializer(serializers.ModelSerializer):
    """ Serializes the HarvestPrediction model objects and return selected fields """

    post_harvest_practices = HarvestPracticeSerializer(source='variety.harvestpractice_set', many=True)

    class Meta:
        model = HarvestPrediction
        fields = '__all__'

