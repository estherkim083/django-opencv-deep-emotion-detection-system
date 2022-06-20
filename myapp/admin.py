from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import PhotoZoneData, ShareEmotion


class PhotoZoneDataAdmin(admin.ModelAdmin):
    search_fields = ['video']


class ShareEmotionAdmin(admin.ModelAdmin):
    search_fields = ['subject']


admin.site.register(PhotoZoneData, PhotoZoneDataAdmin)
admin.site.register(ShareEmotion, ShareEmotionAdmin)
