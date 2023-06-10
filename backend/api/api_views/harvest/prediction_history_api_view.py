from rest_framework import permissions
from rest_framework.generics import ListAPIView

from api.models import HarvestPrediction

from api.serializers import HarvestPredictionSerializer


class HarvestPredictionHistoryAPIView(ListAPIView):
    """ Handles Harvest Prediction History related operations """

    permission_classes = [permissions.IsAuthenticated]

    serializer_class = HarvestPredictionSerializer

    def get_queryset(self):
        # get number of records needed from url query params
        limit = self.request.GET.get('limit')

        if limit:
            return HarvestPrediction.objects.filter(user=self.request.user).order_by('-id')[:int(limit)]
        else:
            # if no prediction limit provided return empty
            return HarvestPrediction.objects.none()




