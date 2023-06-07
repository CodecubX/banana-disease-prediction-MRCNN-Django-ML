from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response

from api.models import test_model


class TestAPIView(APIView):
    """ Run test endpoints """

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        obj = test_model.TestModel.objects.all()[0]
        print(obj)

        context = {
            'text': obj.test
        }

        return Response(context)


