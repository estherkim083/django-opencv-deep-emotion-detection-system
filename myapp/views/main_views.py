import json

from django.shortcuts import render
import tensorflow as tf
from django.utils import timezone
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from django.http import HttpResponseRedirect
import cv2
import numpy as np
import time

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# import mediapipe as mp
#
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

from ..models import PhotoZoneData


def cnn():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(48, 48, 1)
    ))
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.25))
    model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(units=7, activation='softmax'))

    return model


cnn_model = cnn()


def index(request):
    return render(request, "myapp/main.html")


# cnn 모델 학습시키기
def opencv_train(request):
    train_dir = "static/opencv/data/train"
    val_dir = "static/opencv/data/test"
    num_train = 28709
    num_val = 21048
    batch_size = 64
    num_epoch = 50  # 50
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

    cnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

    model_info = cnn_model.fit(
        x=train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size)
    # plot_model_history(model_info)
    cnn_model.save_weights('static/opencv/model.h5')
    print("finished training")
    return HttpResponseRedirect('/myapp/')


# opencv 에서 cnn 모델 파일 로드해서 얼굴 표정 감지 진행
def opencv_display(request):
    cnn_model.load_weights('static/opencv/model.h5')

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
            prediction = cnn_model.predict(cropped_img)
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


def resize_face(face):
    x = tf.expand_dims(tf.convert_to_tensor(face), axis=-1)
    return tf.image.resize(x, (48, 48))


def recognition_preprocessing(faces):
    x = tf.convert_to_tensor([resize_face(f) for f in faces])
    return x


from mtcnn.mtcnn import MTCNN


def mtcnn_opencv_display(request):
    cnn_model.load_weights('static/opencv/model.h5')

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

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

    # start the webcam feed
    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    detector = MTCNN()
    out = cv2.VideoWriter("static/" + timestring + 'output.mp4', fourcc, 7.0, (int(cap.get(3)), int(cap.get(4))),
                          1)  # mp4 filename models
    # out = cv2.VideoWriter('d:/output.mp4', fourcc, 20.0, (640,480))
    while True:
        frames = []
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        location = detector.detect_faces(frame)
        if len(location) > 0:
            faces = []
            pos = []
            for face in location:
                x, y, width, height = face['box']
                x2, y2 = x + width, y + height
                # cv2.rectangle(frame, (x, y), (x2, y2), (255, 20, 147), 4)
                face = frame[y:y2, x:x2]
                # print(face)
                faces.append(face)
                pos.append((x, y, x2, y2))

            x = recognition_preprocessing(faces)

            for i in range(len(faces)):
                try:
                    prediction = cnn_model.predict(x[i])
                    maxindex = int(np.argmax(prediction))
                    # print(maxindex)
                    cv2.rectangle(frame, (pos[i][0], pos[i][3]), (pos[i][2], pos[i][1]), (255, 20, 147), 2)
                    cv2.putText(frame, emotion_dict[maxindex], (pos[i][0], pos[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2, cv2.LINE_AA)
                except KeyError:
                    maxindex = 4
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
                # print(emotion_dict[maxindex])
                emotion_labeled_data += str(maxindex)  # emotion label data models
                # print(emotion_labeled_data)

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

import imutils

def dnn_opencv_display(request):
    cnn_model.load_weights('static/opencv/model.h5')

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

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

    # start the webcam feed
    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    detector = MTCNN()
    out = cv2.VideoWriter("static/" + timestring + 'output.mp4', fourcc, 7.0, (int(cap.get(3)), int(cap.get(4))),
                          1)  # mp4 filename models
    # out = cv2.VideoWriter('d:/output.mp4', fourcc, 20.0, (640,480))
    while True:
        frames = []
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break

        prototextPath = "static/opencv/face_detector/deploy.prototxt"
        caffeModel = "static/opencv/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNetFromCaffe(prototextPath, caffeModel)

        # extract the dimensions , Resize image into 300x300 and converting image into blobFromImage
        (h, w) = frame.shape[:2]
        # blobImage convert RGB (104.0, 177.0, 123.0)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()

        for i in range(0, detections.shape[2]):
            # extract the confidence and prediction

            confidence = detections[0, 0, i, 2]

            # filter detections by confidence greater than the minimum confidence
            if confidence < 0.5 :
                continue
            try:
                # Determine the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                print(confidence)
                # draw the bounding box of the face along with the associated
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                xx= frame[startY: endY, startX: endX]
                xx= recognition_preprocessing(xx)
                prediction = cnn_model.predict(xx)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, emotion_dict[maxindex], (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            except KeyError:
                maxindex = 4
                cv2.putText(frame, emotion_dict[maxindex], (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

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
            emotion_labeled_data += str(maxindex)  # emotion label data models
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)

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
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    return HttpResponseRedirect('/myapp/')
