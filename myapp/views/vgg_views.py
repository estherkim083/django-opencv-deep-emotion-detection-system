import json
import time

import numpy as np
from django.http import HttpResponseRedirect
import cv2
from django.utils import timezone

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation, Dense

from ..models import PhotoZoneData
from keras.preprocessing.image import ImageDataGenerator


# vgg 모델 만들기
def vgg_face():
    # Initialize the model
    model = Sequential()

    # layer 1-2: 2 convolutional layers + 1 max-pooling layer
    model.add(Convolution2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = (48, 48, 1)))
    model.add(Convolution2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = 2))

    # number of filters and convolutions in each layer:
    filters_convs = [(128, 2), (256, 3), (512, 3), (512,3)]

    for n_filters, n_convs in filters_convs:
        for _ in np.arange(n_convs):
            model.add(Convolution2D(filters = n_filters, kernel_size = (3,3), padding = 'same', activation = 'relu'))
        # max-pooling layer
        model.add(MaxPooling2D(pool_size = (2,2), strides = 2))

    model.add(Flatten())
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7,activation = 'softmax'))
    return model

vgg_model = vgg_face()

# vgg 모델 학습시키기
def vgg_opencv_train(request):
    train_dir = "static/opencv/data/train"
    val_dir = "static/opencv/data/test"
    classes = ('angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised')
    # vgg_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

    num_train = 28709
    num_val = 21048
    batch_size = 64
    num_epoch = 35  # 50
    steps_per_epoch = num_train // batch_size
    # validation_steps = num_val//batch_size
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

    vgg_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

    model_info = vgg_model.fit(
        x=train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size)
    # plot_model_history(model_info)
    vgg_model.save_weights('static/opencv/vgg_model.h5')
    print("finished training")
    return HttpResponseRedirect('/myapp/')


# opencv 에서 vgg 모델 파일 로드해서 얼굴 표정 감지 진행
def opencv_vgg_display(request):
    vgg_model.load_weights('static/opencv/vgg_model.h5')

    eng_month = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10,
                 "Nov": 11, "Dec": 12}
    timeclock = time.ctime()
    timeclock_array = timeclock.split()
    month = timeclock_array[1]
    day = timeclock_array[2]
    year = timeclock_array[4]
    month_int = eng_month[month]
    hour = timeclock_array[3].split(":")[0]
    minute = timeclock_array[3].split(":")[1]
    second = timeclock_array[3].split(":")[2]

    created_time = timezone.now()

    timestring = str(year) + "_" + str(month_int) + "_" + str(day) + "_" + str(hour) + "_" + str(minute) + "_" + str(
        second) + "_" + str(request.user) + "_"
    print(timestring)  # datetime models

    emotion_labeled_data = ""
    count = 0
    img_counter = 0
    frame_cnt = 0
    img_name = ""
    images = []
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'X264')

    out = cv2.VideoWriter("static/" + timestring + 'output.mp4', fourcc, 7.0, (int(cap.get(3)), int(cap.get(4))),
                          1)  # mp4 filename models
    # out = cv2.VideoWriter('d:/output.mp4', fourcc, 20.0, (640,480))
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break

        facecasc = cv2.CascadeClassifier("static/opencv/haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 20, 147), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = vgg_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))

            cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2, cv2.LINE_AA)
            frame_cnt += 1
            k = cv2.waitKey(1)
            if k % 256 == 32:
                # SPACE pressed
                img_name = "opencv_frame_{}.png".format(img_counter)
                images.append(timestring + img_name)
                cv2.imwrite("static/" + timestring + img_name, frame)  # image filename models
                print("{} written!".format(img_name))
                img_counter += 1
            out.write(frame)
            print(emotion_dict[maxindex])
            emotion_labeled_data += str(maxindex)  # emotion label data models

        cv2.imshow('Video', cv2.resize(frame, (1000, 740), interpolation=cv2.INTER_CUBIC))  # ??
        if frame_cnt == 30:
            if request.user.is_authenticated:
                photo_images = json.dumps(images)
                photo_zone_data = PhotoZoneData(owner=request.user, datetime=timestring + "datetime",
                                                emotion_labeled_data=emotion_labeled_data,
                                                video=timestring + 'output.mp4', photo=photo_images,
                                                create_date=created_time)
                photo_zone_data.save()
                break
            else:
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return HttpResponseRedirect('/myapp/')
