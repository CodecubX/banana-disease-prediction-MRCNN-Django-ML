from rest_framework import permissions
from rest_framework.generics import ListAPIView

from api.models import DiseasePrediction

from api.serializers import DiseasePredictionSerializer


class DiseasePredictionHistoryAPIView(ListAPIView):
    """ Handles Disease Prediction History related operations """

    permission_classes = [permissions.IsAuthenticated]

    serializer_class = DiseasePredictionSerializer

    def get_queryset(self):
        # get number of records needed from url query params
        limit = self.request.GET.get('limit')

        if limit:
            return DiseasePrediction.objects.filter(user=self.request.user).order_by('-created_at')[:int(limit)]
        else:
            # if no prediction limit provided return empty
            return DiseasePrediction.objects.none()




