from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    camera_stream_proxy,
    AnalysisJobViewSet,
    analysis_index_view,
    analysis_detail_view,
    dashboard_view,
)

router = DefaultRouter()
router.register(r"analyses", AnalysisJobViewSet, basename="analysis-job")

urlpatterns = [
    path("analyses/", analysis_index_view, name="analysis_index"),
    path("analyses/<int:pk>/", analysis_detail_view, name="analysis_detail"),
    path("analyses/dashboard/", dashboard_view, name="analysis_dashboard"),
    path("analyses/camera-stream/", camera_stream_proxy, name="camera_stream_proxy"),
    path("api/v1/", include(router.urls)),
]
