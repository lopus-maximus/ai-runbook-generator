from django.db import models

class MeetingInput(models.Model):
    STATUS_CHOICES = [
        ('uploaded', 'Uploaded'),
        ('processing', 'Processing'),
        ('processed', 'Processed'),
        ('failed', 'Failed')
    ]

    video = models.FileField(upload_to="videos/")
    audio = models.FileField(upload_to="audios/", null=True, blank=True)  
    pdf = models.FileField(upload_to="pdfs/", null=True, blank=True)
    text_input = models.TextField(null=True, blank=True)
    selected_frames = models.JSONField(null=True, blank=True)
    template_schema = models.JSONField(null=True, blank=True)

    transcript = models.TextField(null=True, blank=True)
    detected_language = models.CharField(max_length=10, null=True, blank=True)
    was_translated = models.BooleanField(default=False)

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploaded')
    frame_count = models.IntegerField(default=0)
    processing_time = models.FloatField(default=0.0)  # seconds

    created_at = models.DateTimeField(auto_now_add=True)

class Frame(models.Model):
    meeting = models.ForeignKey(MeetingInput, on_delete=models.CASCADE)
    image = models.ImageField(upload_to="frames/")

class Runbook(models.Model):
    meeting = models.OneToOneField(MeetingInput, on_delete=models.CASCADE)
    content = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)