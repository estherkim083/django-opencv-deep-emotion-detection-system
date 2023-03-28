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
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.backend import sigmoid
from keras.models import Model

# efficient net 분류 학습기를 통해서 기존 데이터셋을 분류/분류된 이미지 저장 작업
# mobile net의 경우, model 생성 부분을 제거하고, load_model을 통해서 mobilenet.h5 파일을 그저 불러와서 적용하면 됨.
# mobilenet 의 경우와 model 생성 부분 밑의 소스 코드는 모두 동일함.(경로 제외)

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

model.load_weights('C:\\Users\\khi\\jupyter\\efficientnet.h5') #efficientnet.h5

count=0
img_counter = 0
frame_cnt=0
# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}



# PIL_img = Image.open("C:\\Users\\khi\\jupyter\\dataset\\1.png") 
train_dir= "C:\\Users\\khi\\jupyter\\data\\train"
test_dir= "C:\\Users\\khi\\jupyter\\data\\test" 
# picture 들을 모두 detect, classify 해서 label 로 분류
label_train_dir= "C:\\Users\\khi\\jupyter\\facial_emotion_detected_img_eff\\train"
label_test_dir= "C:\\Users\\khi\\jupyter\\facial_emotion_detected_img_eff\\test" 

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
    roi = np.resize(img_numpy, (1, 48, 48, 3))
    prediction = model.predict(roi)
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
    roi = np.resize(img_numpy, (1, 48, 48, 3))
    prediction = model.predict(roi)
    maxindex = int(np.argmax(prediction))
    print(emotion_dict[maxindex])

    label_name= emotion_dict[maxindex]
    dirname = os.path.join(label_test_dir, label_name)
    img_name = "opencv_frame_{}_{}.png".format(label_name, img_counter)
    cv2.imwrite(dirname+"\\"+img_name, frame)
    print("{} written!".format(img_name))
    printProgressBar(i + 1, test_index, prefix = 'Progress2:', suffix = 'Complete', length = 50)
    
cv2.destroyAllWindows()