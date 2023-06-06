from django.contrib import admin

from api.models.user_model import User


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    pass
