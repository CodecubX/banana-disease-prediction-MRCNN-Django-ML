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

from api.utils.disease_prediction_mrcnn.predictor import MaskRCNNModel


model = MaskRCNNModel()


class BananaDiseaseMRCNNAPIView(APIView):
    """ Handles Disease Detection using MRCNN model and related operations """

    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

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
        # save copy
        unique_filename = str(uuid.uuid4()) + '.jpg'
        destination_path = os.path.join(destination_directory, unique_filename)

        # Create the destination directory if it doesn't exist
        os.makedirs(destination_directory, exist_ok=True)

        shutil.copyfile(original_img_path, destination_path)

        # get predictions
        predictions = model.predict(destination_path)

        # Construct the URL for the output image
        img_url = request.build_absolute_uri(settings.MEDIA_URL + destination_path)

        # Remove sub-objects with an area of 0
        filtered_data = {}
        for class_name, objects in predictions.items():
            filtered_objects = [obj for obj in objects if obj['area'] != 0]
            if filtered_objects:
                filtered_data[class_name] = filtered_objects

        # Calculate average confidence scores and total area for each class
        result_dict = {}
        for class_name, objects in filtered_data.items():
            avg_confidence = sum(obj['score'] for obj in objects) / len(objects)
            total_area = sum(obj['area'] for obj in objects)
            result_dict[class_name] = {'avg_confidence': avg_confidence, 'total_area': total_area}

        # Sort the result dictionary based on total area in descending order
        sorted_result = dict(sorted(result_dict.items(), key=lambda x: x[1]['total_area'], reverse=True))

        context = {
            'predictions': sorted_result,
            'img_url': img_url
        }

        return Response(context)
