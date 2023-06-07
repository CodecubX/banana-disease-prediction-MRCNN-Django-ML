from django.urls import path
from .api_views import CreateAccount, CurrentUser, HarvestPredictionAPIView

app_name = 'api'

urlpatterns = [
   path('profile/create', CreateAccount.as_view(), name="create_user"),
   path('profile', CurrentUser.as_view(), name="profile"),
   # Harvest predictions related paths
   path('harvest/predict', HarvestPredictionAPIView.as_view(), name='harvest_prediction'),

]