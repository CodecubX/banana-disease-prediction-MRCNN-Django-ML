from rest_framework import permissions
from rest_framework.generics import RetrieveAPIView, get_object_or_404

from api.models import Disease

from api.serializers import DiseaseCureSerializer


class DiseaseAndCureAPIView(RetrieveAPIView):
    """ Handles Disease and their Cures related operations """

    permission_classes = [permissions.IsAuthenticated]

    queryset = Disease.objects.all()
    serializer_class = DiseaseCureSerializer
    lookup_field = 'name'

    def get_object(self):
        queryset = self.filter_queryset(self.get_queryset())
        filter_kwargs = {self.lookup_field: self.request.query_params.get(self.lookup_field)}
        obj = get_object_or_404(queryset, **filter_kwargs)
        self.check_object_permissions(self.request, obj)
        return obj





