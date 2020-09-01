from keras.preprocessing import image
#from keras.models import load_model
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import pygame.mixer
import numpy as np
import picamera
from PIL import Image
from time import sleep
import efficientnet.tfkeras as efn
import time
import tensorflow as tf
from tensorflow.keras import layers as L
import sys
import cv2 

Threshold=0.285 #学習時における検証用予測結果の2σを閾値とする
        
#model building
IMAGE_SIZE = [384,384]

#model作成（重みと同じモデル）
model = tf.keras.Sequential([
    efn.EfficientNetB5(
        input_shape=(*IMAGE_SIZE, 3),
        weights='imagenet',
        include_top=False
    ),
    L.GlobalAveragePooling2D(),
    L.Dense(1024, activation = 'relu'), 
    L.Dropout(0.3), 
    L.Dense(512, activation= 'relu'), 
    L.Dropout(0.2), 
    L.Dense(256, activation='relu'), 
    L.Dropout(0.2), 
    L.Dense(128, activation='relu'), 
    L.Dropout(0.1), 
    L.Dense(1, activation='sigmoid')
])  

if __name__ == '__main__':
    model.load_weights('models/complete_data_efficient_weights.h5') # 重みを読込み
    #model=load_model("models/complete_data_efficient_model.h5") #モデル+重みを使用できる場合はこちらを利用
        
    cap=cv2.VideoCapture(0)
    while True:
	#カメラに写っているものを表示
        rr,img=cap.read()      
        cv2.imshow("img",img)
        key=cv2.waitKey(1)&0xff
        
        if key==ord("p"): #pを押して撮影
            cv2.imwrite("pic.jpg",img) #画像の取得
            photo_filename="pic.jpg"
            img = photo_filename
            img = img_to_array(load_img(img, target_size=(384,384)))
            img_nad = img_to_array(img)/255
            img_nad = img_nad[None, ...]

            pred = model.predict(img_nad, batch_size=1, verbose=0) #予測
            score = np.max(pred)
            print(round(score,3),"Malignant" if score>Threshold else "Benign")
            if key==27: #Escを押して終了
                break

    cap.release()


    cv2.destroyAllWindows()

