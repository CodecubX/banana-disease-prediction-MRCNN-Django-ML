from rest_framework import serializers

from api.models.fertilizer_plan import FertilizerPlan


class FertilizerPlanSerializer(serializers.ModelSerializer):
    """ Serializes the FertilizerPlan model objects and return selected fields """

    fertilizer_type = serializers.CharField(source='get_fertilizer_type_display')
    stage = serializers.CharField(source='get_stage_display')
    variety = serializers.StringRelatedField()

    class Meta:
        model = FertilizerPlan
        fields = '__all__'
