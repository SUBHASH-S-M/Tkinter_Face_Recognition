import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image
import tempfile
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
import smtplib
from gtts import gTTS 
import os
import numpy

import imutils #resize reshaping cropping fedd from camera

import time
import os #to handle directories

algorithm=r"C:\Users\91759\Desktop\project-set-2\pantech\haarcascade_frontalface_default.xml"

haarcscade=cv2.CascadeClassifier(algorithm)
dataset="classifierdataset"


window=tk.Tk()
window.title("Face Recognition System")



l1=tk.Label(window,text="Name",font=("Algerian",20))
l1.grid(column=0, row=0)
t1=tk.Entry(window,width=50,bd=5)
t1.grid(column=1, row=0)

l2=tk.Label(window,text="Age",font=("Algerian",20))
l2.grid(column=0, row=2)
t2=tk.Entry(window,width=50,bd=5)
t2.grid(column=1, row=2)

l3=tk.Label(window,text="Address",font=("Algerian",20))
l3.grid(column=0, row=3)
t3=tk.Entry(window,width=50,bd=5)
t3.grid(column=1, row=3)

def generate_dataset():
    
    dataset="classifierdataset"
    name=t1.get()
    path=os.path.join(dataset,name)
    if not os.path.isdir(path):  #checking a directory exist or not
        os.makedirs(path)
    
    cam=cv2.VideoCapture(0)
    (width,height)=(130,100)
    count=1
    while count<31:
        
        _,img=cam.read()
        gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=haarcscade.detectMultiScale(gray_image,1.3,4) #detdct co-ordinates
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            onlyFace=gray_image[y:y+h,x:x+w]
            resize_img=cv2.resize(onlyFace,(width,height))
            cv2.imwrite("%s/%s.jpg"%(path,count),resize_img)
            count+=1
        cv2.imshow("Face_detection",img)
        key=cv2.waitKey(10)
        
        if key==27:
             break
    print("face_captured")
    cam.release()    
    cv2.destroyAllWindows()

def  Detect_face():
    
    
      #subh,name-labels
    #imasges name 1.jpg 2.jpg is names
    #id(0) 1st flder name
    (images,labels,names,id)=([],[],{},0)
    #subdirs is named as classiferdatsets
    #dirs is folder named as subhash etc
    #files is named as 1,2,3,4,5.jpg
    for (subdirs,dirs,files) in os.walk(dataset):
        for subdirs in dirs:
            names[id]=subdirs
            subjectpath=os.path.join(dataset,subdirs)
            for filename in os.listdir(subjectpath):
                paths=subjectpath+'/'+filename
                label=id
                images.append(cv2.imread(paths,0))
                labels.append(int(label))
            id+=1
    (width,height)=(130,100)
    (images,labels)=[numpy.array(lis) for lis in [images,labels] ]
    print(images,labels)            
    
    #30 images array plus label 0 1
    
    
    # Model Classifier
    
    model=cv2.face.LBPHFaceRecognizer_create()
    #model=cv2.face.FisherFaceRecognizer_create()
    
    model.train(images,labels)
    print("training completed")


     # Model Classifier
    
    model=cv2.face.LBPHFaceRecognizer_create()
    #model=cv2.face.FisherFaceRecognizer_create()
    
    model.train(images,labels)
    print("training completed")

    cam=cv2.VideoCapture(0)
    (width,height)=(130,100)
    count=1
    while count<31:
        
        _,img=cam.read()
        gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=haarcscade.detectMultiScale(gray_image,1.3,4) #detdct co-ordinates
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            onlyFace=gray_image[y:y+h,x:x+w]
            resize_img=cv2.resize(onlyFace,(width,height))
            prediction=model.predict(resize_img)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            if prediction[1]<800:
                cv2.putText(img,"%s - %.0f"%(names[prediction[0]],prediction[1]),
                           (x-20,y-10),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0))
                print(names[prediction[0]])
                count=0
            else:
                count+=1
                cv2.putText(img,"Unknown",
                           (x-20,y-10),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0))
                if(count>100):
                    print("Unknown Person")
                    cv2.imwrite("Input.jpg",img)
                    count=0
        cv2.imshow("OpenCV",img)
        key=cv2.waitKey(10)
        if key==27:
            break
    cam.release()
    cv2.destroyAllWindows()















      
b2=tk.Button(window,text="Detect the face",font=("Algerian",17),bg='green',fg='white',command=Detect_face)
b2.grid(column=2, row=4)




b3=tk.Button(window,text="Generating dataset",font=("Algerian",17),bg='green',fg='white',command=generate_dataset)
b3.grid(column=1, row=4)


window.geometry("800x200")
window.mainloop()
