from django.urls import path
from .api_views import CreateAccount, CurrentUser, HarvestPredictionAPIView, TestAPIView, VarietyListAPIView, \
   VarietyRetrieveAPIView, HarvestPredictionHistoryAPIView, BananaDiseaseMRCNNAPIView

app_name = 'api'

urlpatterns = [
   path('profile/create', CreateAccount.as_view(), name="create_user"),
   path('profile', CurrentUser.as_view(), name="profile"),

   # variety related paths
   path('banana-variety/all', VarietyListAPIView.as_view(), name='variety-list'),
   path('banana-variety/<int:pk>', VarietyRetrieveAPIView.as_view(), name='variety-retrieve'),

   # Harvest predictions related paths
   path('harvest/predict', HarvestPredictionAPIView.as_view(), name='harvest_prediction'),
   path('harvest/data', HarvestPredictionAPIView.as_view(), name='harvesting_time'),
   path('harvest/prediction-history', HarvestPredictionHistoryAPIView.as_view(), name='harvest_prediction_history'),

   # Image based disease detection - MRCNN
   path('disease/detect', BananaDiseaseMRCNNAPIView.as_view(), name='disease_detection_mrcnn'),

   # --- for testing --
   path('tests', TestAPIView.as_view(), name='test')
]
