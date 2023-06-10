from rest_framework import serializers

from api.models.harvest.harvest_practice_model import HarvestPractice


class HarvestPracticeSerializer(serializers.ModelSerializer):
    """ Serializes the HarvestPractice model objects and return selected fields """

    class Meta:
        model = HarvestPractice
        fields = ['practice_name', 'description']