from django.contrib import admin

from api.models import user_model, test_model, variety_model, harvest_prediction_model, harvest_practices_model


# --- for testing ---
@admin.register(test_model.Test)
class TestAdmin(admin.ModelAdmin):
    pass


@admin.register(user_model.User)
class UserAdmin(admin.ModelAdmin):
    pass


@admin.register(variety_model.Variety)
class VarietyAdmin(admin.ModelAdmin):
    pass


@admin.register(harvest_prediction_model.HarvestPrediction)
class HarvestPredictionAdmin(admin.ModelAdmin):
    pass


@admin.register(harvest_practices_model.HarvestPractice)
class HarvestPracticeAdmin(admin.ModelAdmin):
    pass


