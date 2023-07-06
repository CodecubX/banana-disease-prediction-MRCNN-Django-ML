from rest_framework import serializers

from api.models import Cure


class CureSerializer(serializers.ModelSerializer):
    """ Serializes the Cure model objects """

    name_display = serializers.CharField(source='name')
    img = serializers.SerializerMethodField()

    class Meta:
        model = Cure
        fields = ['id', 'name_display', 'description', 'img', 'disease']

    def get_img(self, obj):
        if obj.img:
            return self.context['request'].build_absolute_uri(obj.img.url)
        return None


class CureSinhalaSerializer(serializers.ModelSerializer):
    """ Serializes the Cure model objects """

    name_display = serializers.CharField(source='name_sinhala')
    description = serializers.CharField(source='description_sinhala', allow_blank=True)
    img = serializers.SerializerMethodField()

    class Meta:
        model = Cure
        fields = ['id', 'name_display', 'description', 'img', 'disease']

    def get_img(self, obj):
        if obj.img:
            return self.context['request'].build_absolute_uri(obj.img.url)
        return None