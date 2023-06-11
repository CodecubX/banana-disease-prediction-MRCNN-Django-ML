import os
import shutil
import uuid
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rest_framework import permissions, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from api.models import Disease, DiseasePrediction

from api.utils.disease_prediction_mrcnn.predictor import MaskRCNNModel

from api.serializers import DiseaseCureSerializer

from .utils.utils import filter_and_calculate_area

model = MaskRCNNModel()


class BananaDiseaseMRCNNAPIView(APIView):
    """ Handles Disease Detection using MRCNN model and related operations """

    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    serializer_class = DiseaseCureSerializer

    def post(self, request, *args, **kwargs):
        try:
            file_obj = request.FILES['image']
        except Exception:
            return Response(
                {'error': 'Something is wrong with the uploaded file'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # save the image
        original_img_file = default_storage.save('disease_detection/original/temp_image.jpg', ContentFile(file_obj.read()))
        original_img_path = os.path.join(settings.MEDIA_ROOT, original_img_file)

        destination_directory = os.path.join(settings.MEDIA_ROOT, 'disease_detection', 'detected')
        # create unique file name
        unique_filename = str(uuid.uuid4()) + '.jpg'

        destination_path = os.path.join(destination_directory, unique_filename)

        # create the destination directory if it doesn't exist
        os.makedirs(destination_directory, exist_ok=True)
        # save copy
        shutil.copyfile(original_img_path, destination_path)

        # get predictions
        predictions = model.predict(destination_path)

        # Construct the URLs for the output images
        original_img_url = request.build_absolute_uri(settings.MEDIA_URL + original_img_path)
        detected_img_url = request.build_absolute_uri(settings.MEDIA_URL + destination_path)

        # Sort the result dictionary based on total area in descending order
        sorted_results = filter_and_calculate_area(predictions)

        top_disease = list(sorted_results.keys())[0]

        try:
            disease = Disease.objects.filter(name=top_disease).first()
            disease = self.serializer_class(disease).data

            try:
                # Create an instance of DiseasePrediction
                prediction = DiseasePrediction()

                # Set the fields with the corresponding values
                prediction.img = original_img_path
                prediction.detected_img = destination_path
                prediction.user = request.user
                prediction.disease = disease
                prediction.top_probabilities = sorted_results

                # Save the instance
                prediction.save()
            except Exception as e:
                print(f'INFO: Failed to save to history: {e}')

        except Disease.DoesNotExist:
            disease = 'Disease not found'

        context = {
            'prediction': disease,
            'top_probabilities': sorted_results,
            'original_img_url': original_img_url,
            'detected_img_url': detected_img_url
        }

        return Response(context)
