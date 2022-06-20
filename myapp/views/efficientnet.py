import json
from django.utils import timezone
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from django.http import HttpResponseRedirect
import cv2
import time

from ..models import PhotoZoneData
import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.backend import sigmoid
from keras.models import Model


class SwishActivation(Activation):

    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'


def swish_act(x, beta=1):
    return (x * sigmoid(beta * x))


from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

IMAGE_SIZE = [48, 48]
model_create = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=IMAGE_SIZE + [3], pooling='avg',
                                             weights='imagenet')

# Adding 2 fully-connected layers to B0.
x = model_create.output

x = BatchNormalization()(x)
x = Dropout(0.7)(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation(swish_act)(x)
x = Dropout(0.5)(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation(swish_act)(x)

# Output layer
predictions = Dense(7, activation="softmax")(x)

model = Model(inputs=model_create.input, outputs=predictions)


# efficientnet 모델 학습시키기
def opencv_efficientnet_train(request):
    train_dir = "static/opencv/data/train"
    val_dir = "static/opencv/data/test"
    epochs = 25
    batch_size = 32
    train_samples = 28709
    validation_samples = 21048
    img_width, img_height = 48, 48

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    # Create a train generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='rgb',
        shuffle=True,
        class_mode='categorical')
    # Create a test generator
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='rgb',
        shuffle=True,
        class_mode='categorical')
    # Start training, fit the model

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

    model.fit(
        x=train_generator,
        steps_per_epoch=train_samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size,
        epochs=epochs)
    # Save model to disk
    model.save('static/opencv/efficientnet.h5')
    print('Saved model to disk!')
    return HttpResponseRedirect('/myapp/')

# efficientnet 모델 이어서 학습시키기
def efficientnet_train_again(request):
    train_dir = "static/opencv/data/train"
    val_dir = "static/opencv/data/test"

    epochs = 1
    batch_size = 32
    train_samples = 28709
    validation_samples = 21048
    img_width, img_height = 48, 48

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    # Create a train generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='rgb',
        shuffle=True,
        class_mode='categorical')
    # Create a test generator
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='rgb',
        shuffle=True,
        class_mode='categorical')

    model.load_weights('static/opencv/efficientnet.h5')
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
    model.summary()
    model.fit(
        x=train_generator,
        steps_per_epoch=train_samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size,
        epochs=epochs)

    model.save('static/opencv/efficientnet.h5')
    print('Saved model to disk!')
    return HttpResponseRedirect('/myapp/')

# opencv 에서 efficientnet 모델 파일 로드해서 얼굴 표정 감지 진행
def opencv_efficientnet_display(request):
    model.load_weights('static/opencv/efficientnet.h5')

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
            # roi_gray = gray[y:y + h, x:x + w]
            # cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

            roi_color = frame[y:y + h, x:x + w]
            # recognizing emotion
            roi = np.array(roi_color)
            roi = np.resize(roi, (1, 48, 48, 3))
            prediction = model.predict(roi)

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
