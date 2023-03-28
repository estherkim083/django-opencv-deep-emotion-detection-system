from random import *
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from turtle import window_height
import numpy as np
import argparse
import matplotlib.pyplot as plt
from cv2 import *
import cv2
import imutils
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation, Dense

#cnn model, vgg model 기존 이미지 데이터셋 분류시킬 때 필요한 코드.
#단, cnn model은 vgg_face() 대신에 django file src/myapp/views/main_view.py에 있는 cnn model 생성 코드로 대체해서 실행하면 됨.
#모델(.h5) 파일 불러오는 경로만 다르고 나머지 코드는 cnn , vgg의 경우 모두 동일함.

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

model= vgg_face()

model.load_weights("C:\\Users\\khi\\jupyter\\vgg_model.h5") 

count=0
img_counter = 0
frame_cnt=0
# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}



#PIL_img = Image.open("C:\\Users\\khi\\jupyter\\dataset\\1.png") 
train_dir= "C:\\Users\\khi\\jupyter\\data\\train"
test_dir= "C:\\Users\\khi\\jupyter\\data\\test" 
# picture 들을 모두 detect, classify 해서 label 로 분류
label_train_dir= "C:\\Users\\khi\\jupyter\\facial_emotion_detected_img_vgg\\train"
label_test_dir= "C:\\Users\\khi\\jupyter\\facial_emotion_detected_img_vgg\\test" 

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
def get_subdir(folder):
    listDir = None
    for root, dirs, files in os.walk(folder):
        if not dirs == []:
            listDir = dirs
            break
    listDir.sort()
    return listDir

def list_of_img_files(folder, number=0):
    filelists = []
    subdir = get_subdir(folder)
    for label in range(0, len(subdir)):
        filelist = []
        filelists.append(filelist)
        dirname = os.path.join(folder, subdir[label])
        for file in os.listdir(dirname):
            if (file.endswith('.png') or file.endswith('.jpg')):
                fullname = os.path.join(dirname, file)
                if (os.path.getsize(fullname) > 0):
                    filelist.append(fullname)
                else:
                    print('file ' + fullname + ' is empty')
        # sort each list of files so they start off in the same order
        # regardless of how the order the OS returns them in
        filelist.sort()
        
    labelsAndFiles = []
    for label in range(0, len(subdir)):
        count = number if number > 0 else len(filelists[label])
        filelist = random.sample(filelists[label], count)
        for filename in filelist:
            labelsAndFiles.append(filename)

    return labelsAndFiles

train_dir_img_files =list_of_img_files(train_dir)
print(len(train_dir_img_files)) #28709
test_dir_img_files= list_of_img_files(test_dir)
print(len(test_dir_img_files)) #21048
img_counter=0
train_index= len(train_dir_img_files)
printProgressBar(0, train_index, prefix = 'Progress:', suffix = 'Complete', length = 50)
for i in range(0, train_index):
    img_counter+=1
    path_for_each_img= train_dir_img_files[i]
    path_for_each_img = path_for_each_img.replace('\\', '\\\\')
    print(path_for_each_img)
    
    frame= cv2.imread(path_for_each_img, 1)
    PIL_img = Image.open(path_for_each_img) 
    img_numpy = np.array(PIL_img)
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(img_numpy, (48, 48)), -1), 0)
    prediction = model.predict(cropped_img)
    maxindex = int(np.argmax(prediction))
    print(emotion_dict[maxindex])

    label_name= emotion_dict[maxindex]
    dirname = os.path.join(label_train_dir, label_name)
    img_name = "opencv_frame_{}_{}.png".format(label_name, img_counter)
    cv2.imwrite(dirname+"\\"+img_name, frame)
    print("{} written!".format(img_name))
    # Update Progress Bar
    printProgressBar(i + 1, train_index, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
img_counter=0
test_index= len(test_dir_img_files)
printProgressBar(0, test_index, prefix = 'Progress2:', suffix = 'Complete', length = 50)
for i in range(0, test_index):
    img_counter+=1
    path_for_each_img= test_dir_img_files[i]
    path_for_each_img = path_for_each_img.replace('\\', '\\\\')
    print(path_for_each_img)
    
    frame= cv2.imread(path_for_each_img, 1)
    PIL_img = Image.open(path_for_each_img) 
    img_numpy = np.array(PIL_img)
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(img_numpy, (48, 48)), -1), 0)
    prediction = model.predict(cropped_img)
    maxindex = int(np.argmax(prediction))
    print(emotion_dict[maxindex])

    label_name= emotion_dict[maxindex]
    dirname = os.path.join(label_test_dir, label_name)
    img_name = "opencv_frame_{}_{}.png".format(label_name, img_counter)
    cv2.imwrite(dirname+"\\"+img_name, frame)
    print("{} written!".format(img_name))
    printProgressBar(i + 1, test_index, prefix = 'Progress2:', suffix = 'Complete', length = 50)
    
cv2.destroyAllWindows()