from django.urls import path, re_path
from .views import main_views, record_views, share_views, vgg_views, efficientnet, mobilenet

app_name= 'myapp'
urlpatterns= [
    path('', main_views.index, name='index'),
    path('photozone/cnn_training', main_views.opencv_train, name='training'),
    path('photozone/vgg_training', vgg_views.vgg_opencv_train, name='vgg-training'),
    path('photozone/opencv_vgg_display', vgg_views.opencv_vgg_display, name='vgg-photozone'),
    path('photozone/efficientnet_training', efficientnet.opencv_efficientnet_train, name='efficientnet-training'),
    path('photozone/efficientnet_train_again', efficientnet.efficientnet_train_again, name='efficientnet-train-again'),
    path('photozone/opencv_efficientnet_display', efficientnet.opencv_efficientnet_display, name='efficientnet-photozone'),
    path('photozone/mobilenet_training', mobilenet.opencv_mobilenet_train, name='mobilenet-training'),
    path('photozone/mobilenet_train_again', mobilenet.mobilenet_train_again, name='mobilenet-train-again'),
    path('photozone/opencv_mobilenet_display', mobilenet.opencv_mobilenet_display, name='mobilenet-photozone'),
    path('photozone', main_views.opencv_display, name='photozone'),
    path('photozone/dnn', main_views.dnn_opencv_display, name='dnn-photozone'),
    path('photozone/mtcnn', main_views.mtcnn_opencv_display, name='mtcnn-photozone'),
    path('record', record_views.record_index, name= "record"),
    path('record/statistics', record_views.statistics, name= "statistics"),
    path('record/videolink/<str:video>', record_views.video_cv_open, name= "video"),
    path('share/write/<str:video>', share_views.share_index, name='share-commit'),
    path('share/record', share_views.share_record, name='share-record'),
    path('share/detail/<str:share_emotion_id>', share_views.share_detail, name='share-detail')
]