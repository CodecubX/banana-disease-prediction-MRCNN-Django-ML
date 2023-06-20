from rest_framework import permissions
from rest_framework.generics import ListAPIView

from api.models import FertilizerPlanPrediction

from api.serializers import FertilizerPlanPredictionSerializer


class FertilizerPlanPredictionHistoryAPIView(ListAPIView):
    """ Handles Fertilizer Plan Prediction History related operations """

    permission_classes = [permissions.IsAuthenticated]

    serializer_class = FertilizerPlanPredictionSerializer

    def get_queryset(self):
        # get number of records needed from url query params
        limit = self.request.GET.get('limit')
        if limit:
            return FertilizerPlanPrediction.objects.filter(user=self.request.user).order_by('-created_at')[:int(limit)]
        else:
            # if no prediction limit provided return empty
            return FertilizerPlanPrediction.objects.none()




