from django.contrib import admin

from api.models import Test, User, Variety, HarvestPractice, HarvestPrediction, Disease, Cure, DiseasePrediction,\
    WateringPlan, WateringPlanPrediction, FertilizerPlan, FertilizerPlanPrediction


# --- for testing ---
@admin.register(Test)
class TestAdmin(admin.ModelAdmin):
    pass


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    pass


@admin.register(Variety)
class VarietyAdmin(admin.ModelAdmin):
    pass


@admin.register(HarvestPrediction)
class HarvestPredictionAdmin(admin.ModelAdmin):
    pass


@admin.register(HarvestPractice)
class HarvestPracticeAdmin(admin.ModelAdmin):
    pass


@admin.register(Disease)
class DiseaseAdmin(admin.ModelAdmin):
    pass


@admin.register(Cure)
class CureAdmin(admin.ModelAdmin):
    pass


@admin.register(DiseasePrediction)
class DiseasePredictionAdmin(admin.ModelAdmin):
    pass


@admin.register(WateringPlan)
class WaterPlanAdmin(admin.ModelAdmin):
    pass


@admin.register(WateringPlanPrediction)
class WaterPlanPredictionAdmin(admin.ModelAdmin):
    pass


@admin.register(FertilizerPlan)
class FertilizerAdmin(admin.ModelAdmin):
    pass


@admin.register(FertilizerPlanPrediction)
class FertilizerPlanPredictionAdmin(admin.ModelAdmin):
    pass
