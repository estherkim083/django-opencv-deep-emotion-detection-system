from django.db import models
from django.contrib.auth.models import User

# Create your models here.
from django.utils import timezone


class InitiateTrainingModel(models.Model):
    count = models.IntegerField(default=0)


class PhotoZoneData(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='photozone_data_owner')
    datetime = models.TextField()
    emotion_labeled_data = models.TextField()  # neutral happy happy -> '433'
    video = models.TextField()
    photo = models.TextField()
    create_date = models.DateTimeField(default=timezone.now())

    def __str__(self):
        return self.video


class ShareEmotion(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='author_question')
    subject = models.CharField(max_length=200)
    content = models.TextField()
    create_date = models.DateTimeField(default=timezone.now())
    modify_date = models.DateTimeField(null=True, blank=True)
    share_emotion = models.ForeignKey(PhotoZoneData, null=True, blank=True, on_delete=models.CASCADE)

    def __str__(self):
        return self.subject


class Comment(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    create_date = models.DateTimeField()
    modify_date = models.DateTimeField(null=True, blank=True)
    shareEmotion = models.ForeignKey(ShareEmotion, null=True, blank=True, on_delete=models.CASCADE)
