from django.contrib import admin
from .models import AnalysisJob, PersonPrediction


class PersonPredictionInline(admin.TabularInline):
    model = PersonPrediction
    extra = 0
    readonly_fields = ("track_id", "action_label", "confidence", "valid_frames", "created_at")
    can_delete = False


@admin.register(AnalysisJob)
class AnalysisJobAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "title",
        "user",
        "input_type",
        "status",
        "created_at",
        "completed_at",
    )
    list_filter = ("status", "input_type", "created_at")
    search_fields = ("title", "user__username", "task_id")
    readonly_fields = ("created_at", "completed_at", "task_id")
    inlines = [PersonPredictionInline]


@admin.register(PersonPrediction)
class PersonPredictionAdmin(admin.ModelAdmin):
    list_display = ("id", "job", "track_id", "action_label", "confidence", "valid_frames", "created_at")
    list_filter = ("action_label", "created_at")
    search_fields = ("action_label", "job__title")
    readonly_fields = ("created_at",)
