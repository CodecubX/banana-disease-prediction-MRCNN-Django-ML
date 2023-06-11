from rest_framework import serializers

from api.models import Disease

from .cure_serializer import CureSerializer


class DiseaseCureSerializer(serializers.ModelSerializer):
    """ Serializes the Disease model objects """

    cures = CureSerializer(source='cure_set', many=True)  # Nested serialization for associated cures
    img = serializers.SerializerMethodField()

    class Meta:
        model = Disease
        fields = ['name', 'description', 'img', 'cures']

    def get_img(self, obj):
        if obj.img:
            return self.context['request'].build_absolute_uri(obj.img.url)
        return None