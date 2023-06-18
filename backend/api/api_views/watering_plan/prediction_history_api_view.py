from rest_framework import permissions
from rest_framework.generics import ListAPIView

from api.models import WateringPlanPrediction

from api.serializers import WateringPlanPredictionSerializer


class WateringPlanPredictionHistoryAPIView(ListAPIView):
    """ Handles Watering Plan Prediction History related operations """

    permission_classes = [permissions.IsAuthenticated]

    serializer_class = WateringPlanPredictionSerializer

    def get_queryset(self):
        # get number of records needed from url query params
        limit = self.request.GET.get('limit')
        if limit:
            return WateringPlanPrediction.objects.filter(user=self.request.user).order_by('-created_at')[:int(limit)]
        else:
            # if no prediction limit provided return empty
            return WateringPlanPrediction.objects.none()




