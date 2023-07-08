from rest_framework import serializers

from api.models import DiseaseQuestionnairePrediction

from .disease_cure_serializer import DiseaseCureSerializer, DiseaseCureSinhalaSerializer


class DiseaseQuestionnairePredictionSerializer(serializers.ModelSerializer):
    """ Serializes the Disease Questionnaire model objects and return selected fields """

    disease = DiseaseCureSerializer()

    class Meta:
        model = DiseaseQuestionnairePrediction
        fields = '__all__'


class DiseaseQuestionnairePredictionSinhalaSerializer(serializers.ModelSerializer):
    """ Serializes the Disease Questionnaire model objects and return selected fields """

    disease = DiseaseCureSinhalaSerializer()

    class Meta:
        model = DiseaseQuestionnairePrediction
        fields = '__all__'