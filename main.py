import cv2
import os
from sklearn.preprocessing import LabelBinarizer as lb
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import scipy.ndimage


def extract_character(image, rec = 0):
      
    dim = (min(image.shape[1],900), min(image.shape[0],900))
    thresh = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)  
    thresh=cv2.GaussianBlur(thresh, (3,3), 0)

    _,thresh=cv2.threshold(thresh,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    kernel1 = np.ones((3,3), dtype= np.uint8)
    thresh = cv2.dilate(thresh, kernel1, iterations = 1)
    cv2.imshow('thresh',thresh)
    cv2.waitKey(0)


    if(rec<2):
    	thresh2 = cv2.erode(thresh, np.ones((2,2), dtype= np.uint8), iterations = 2)
    	thresh2 = scipy.ndimage.median_filter(thresh2, (5, 1)) # remove line noise
    	thresh2 = scipy.ndimage.median_filter(thresh2, (1, 5)) # weaken circle noise
    	thresh2 = scipy.ndimage.median_filter(thresh2, (5, 1)) # remove line noise
    	thresh2 = scipy.ndimage.median_filter(thresh2, (1, 5)) # weaken circle noise
    	contours1, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    else:
    	contours1, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 
    # print(len(contours1))

    contours1 = sorted(contours1, key=lambda x:cv2.contourArea(x), reverse = True)
    if(rec<2):
        cv2.imshow('thresh2',thresh2)
        cv2.waitKey(0)

    meanarea = 0
    for cnt in contours1:
        meanarea= meanarea+cv2.contourArea(cnt)
    meanarea=meanarea/len(contours1)
    area_ratio=cv2.contourArea(contours1[0])/meanarea

    
    # print('arearatio:',area_ratio)
    if(area_ratio>6 and rec<1):
    	kernel2 = np.ones((3,3), np.uint8)
    else:
    	kernel2 = np.ones((2,2), np.uint8)
    # kernel2 = np.ones((2,2), np.uint8)
    if(area_ratio >= 1 and rec<2):
    	thresh = cv2.erode(thresh, kernel2, iterations = 2)
    	thresh = scipy.ndimage.median_filter(thresh, (5, 1)) 
    	thresh = scipy.ndimage.median_filter(thresh, (1, 5)) 
    	thresh = scipy.ndimage.median_filter(thresh, (5, 1)) 
    	thresh = scipy.ndimage.median_filter(thresh, (1, 5))
    if (area_ratio>6 and rec<1):
        thresh = cv2.dilate(thresh, np.ones((2,2), dtype= np.uint8), iterations = 1)
    
    thresh = cv2.dilate(thresh, np.ones((2,2), dtype= np.uint8), iterations = 1)
    
    cv2.imshow('thresh',thresh)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse = True)
    # print(len(contours))
    for i in range(len(contours)):
        (_,_,w1,_)=cv2.boundingRect(contours[i])
        minLineLength= 0.75*w1
        # print(minLineLength)
        lines = cv2.HoughLinesP(thresh,1,np.pi/180,100,minLineLength,10)
        

        # print(lines[0])
        for x1,y1,x2,y2 in lines[0]:
            # print(x1,x2,y1,y2)
            if(x2-x1>y2-y1):
                cv2.line(thresh,(x1,y1),(x2,y2),(0,0,0),int(y1-y2)+20)
                cv2.line(thresh,(x1,y2),(x2,y1),(0,0,0),int(y1-y2)+20)
                cv2.line(thresh,(x1,y2),(x2,y2),(0,0,0),int(y1-y2)+20)
                # cv2.line(thresh,(x1,y2),(x2,y1),(0,0,0),int(y1-y2)+20)
                # thresh= cv2.erode(thresh, np.ones((2,2), dtype= np.uint8), iterations = 1)
    cv2.imshow('thresh',thresh)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    coords=[]
    c=0
    meanArea=0
    for cnt in contours:
        meanArea= meanArea+cv2.contourArea(cnt)
    meanArea=meanArea/len(contours)
    # print(meanArea)
    for cnt in contours:
        (x,y,w,h)=cv2.boundingRect(cnt)
        ratio=w/h
        # print(cv2.contourArea(cnt))
        if cv2.contourArea(cnt)>0.45*meanArea:
            # print(ratio)
            if ratio > 1.5:
                half_width = int(w / 2)
                # print(half_width)
                coords.append((x-6, y-6, half_width+3, h+3))
                coords.append((x-6 + half_width+3, y-6, half_width+3, h+3))
                c=c+2
            else:  
                coords.append((x-6, y-6, w+6, h+6))
                c=c+1
    coords=sorted(coords,key=lambda x: x[0])
    img_paths=[]
    
    # print(c)
    if(c >12 and rec <3):
    	img_paths_array = extract_character(image, rec + 1)
    	return img_paths_array
    else:
    	for i in range(c):
        	result=filter(thresh[coords[i][1]:coords[i][1]+coords[i][3],coords[i][0]:coords[i][0]+coords[i][2]])
        	filename='char'+str(i)+'.jpeg'
        	cv2.imwrite(filename,cv2.bitwise_not(result))
        	img_paths.append(filename)
    	return np.array(img_paths)


def filter(img):
    # img=cv2.copyMakeBorder(img,29,29,29,29,cv2.BORDER_CONSTANT)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))
    meanArea=0
    for cnt in contours:
        meanArea= meanArea+cv2.contourArea(cnt)
    meanArea=meanArea/len(contours)
    
    nlabels,labels,stats,_=cv2.connectedComponentsWithStats(img,None,None,None,8,cv2.CV_32S)
    result=np.zeros((img.shape),np.uint8)
    # print(nlabels)
    # c=0
    for i in range(nlabels-1):
        
        area = stats[i,cv2.CC_STAT_AREA]
        if area >=0.1*meanArea:
            result[labels==i+1]=255
            # c=c+1
    # print(c)
    
    result = cv2.resize(result,(28,28),cv2.INTER_AREA)
    # cv2.line(result,(2,1),(27,1),(255,255,255),1)
    kernel1 = np.ones((2,2), dtype= np.uint8)
    result = cv2.dilate(result, kernel1, iterations = 1)
    return result



model=load_model('my_model1.h5')

    
def predict():
    #Enter filenames to be tested in image_paths after adding them to this folder
    image_paths=['sampleImage/three-lines1.png']
    for i in image_paths:
        image=cv2.imread(i)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_paths=extract_character(image)
        
        for i in img_paths:
            img=cv2.imread(i,cv2.IMREAD_GRAYSCALE)
            # img = cv2.bitwise_not(img)
            img=np.reshape(img,(28,28,1))/255
            m = [img]
            m=np.array(m)
            k=model.predict(m)
            df=pd.read_csv('trainig1.csv')
            df.drop(columns='Unnamed: 0',inplace=True)
            train_y=df['Key']

            num_classes=train_y.nunique()
            train_y=np.asarray(train_y)
            l_b=lb()
            Y=l_b.fit_transform(train_y)
            some=l_b.inverse_transform(k)
            # print(some)
            label=pd.read_csv("label_consonants_vowels.csv")
            label.drop(columns='Unnamed: 0',inplace=True)
            labeling = dict(zip(label.Key,label.Hindi))

            # labeling=labeling[:,-1]
            print(labeling[some[0]])
            # print()
if __name__=='__main__':
    predict()

