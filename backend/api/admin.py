from django.contrib import admin

from api.models import user_model, test_model


# --- for testing ---
@admin.register(test_model.TestModel)
class TestAdmin(admin.ModelAdmin):
    pass


@admin.register(user_model.User)
class UserAdmin(admin.ModelAdmin):
    pass










