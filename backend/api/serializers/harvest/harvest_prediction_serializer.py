from rest_framework import serializers

from api.models import HarvestPrediction

from .harvest_practice_serializer import HarvestPracticeSerializer


class HarvestPredictionSerializer(serializers.ModelSerializer):
    """ Serializes the HarvestPrediction model objects and return selected fields """

    variety = serializers.StringRelatedField()
    post_harvest_practices = HarvestPracticeSerializer(source='variety.harvestpractice_set', many=True)

    class Meta:
        model = HarvestPrediction
        fields = ['id', 'predicted_harvest', 'agro_climatic_region', 'plant_density',
                  'spacing_between_plants', 'pesticides_used', 'plant_generation',
                  'fertilizer_type', 'soil_pH', 'amount_of_sunlight_received',
                  'watering_schedule', 'number_of_leaves', 'height', 'variety',
                  'harvest', 'top_probabilities', 'created_at',
                  'post_harvest_practices', 'post_harvest_practices']

