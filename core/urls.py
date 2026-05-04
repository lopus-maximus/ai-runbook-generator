from django.urls import path
from .views import upload, frame_preview, success, download_transcript, edit_runbook, qa, qa_page

urlpatterns = [
    path('', upload, name="upload"),
    path('frames/<int:meeting_id>/', frame_preview, name="frames"),
    path('success/<int:meeting_id>/', success, name="success"),
    path('transcript/<int:meeting_id>/', download_transcript, name="download_transcript"),
    path('edit/<int:meeting_id>/', edit_runbook, name="edit_runbook"),
    path('qa/<int:meeting_id>/', qa_page, name="qa_page"),
    path('qa/<int:meeting_id>/ask/', qa, name="qa_ask"),
]