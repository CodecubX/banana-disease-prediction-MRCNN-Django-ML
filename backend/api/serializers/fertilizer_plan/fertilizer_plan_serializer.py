from rest_framework import serializers

from api.models.fertilizer_plan import FertilizerPlan


class FertilizerPlanSerializer(serializers.ModelSerializer):
    """ Serializes the FertilizerPlan model objects and return selected fields """

    class Meta:
        model = FertilizerPlan
        fields = '__all__'
