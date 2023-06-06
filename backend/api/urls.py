from django.urls import path
from .api_views import CreateAccount, CurrentUser

app_name = 'api'

urlpatterns = [
   path('profile/create', CreateAccount.as_view(), name="create_user"),
   path('profile', CurrentUser.as_view(), name="profile"),
]