import os
import cv2
import numpy as np

all_files = os.listdir("train_images3")
print(all_files)
txt_files = filter(lambda x: x[-4:] == '.txt', all_files)
print(txt_files)

images = filter(lambda x: x[-4:] == '.jpg', all_files)
print(images)

#y = np.zeros((1,14))
#print (y)
X = []
y= []
for txt in txt_files:
    d="train_images3/"+txt
    f = open(d,"r");
    lines = f.readlines();
    for i in lines:
        thisline = i.split("	");
        print (thisline[1])
        l = thisline[1]
        
        #y=np.array(y)
        if l=="0":
            label = [1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        if l=="1":
            label = [0,1,0,0,0,0,0,0,0,0,0,0,0,0]
        if l=="2":
            label = [0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        if l=="3":
            label = [0,0,0,1,0,0,0,0,0,0,0,0,0,0]        
        if l=="4":
            label = [0,0,0,0,1,0,0,0,0,0,0,0,0,0]               
        if l=="5":
            label = [0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        if l=="6":
            label = [0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        if l=="7":
            label = [0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        if l=="8":
            label = [0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        if l=="9":
            label = [0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        if l=="10":
            label = [0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        if l=="11":
            label = [0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif l=="12" or l=="13":
            continue

            
            '''
        if l=="12":
            label = [0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        if l=="13":
            label = [0,0,0,0,0,0,0,0,0,0,0,0,0,1]
             '''                      
        print (label)
        label=np.array([label])
        y.append(l)
        img = d.split('.')
        
        img=img[0]
        ext=".jpg"
        img=img+ext
        print(img)
        image = cv2.imread(img,0)
        X.append([np.array(image)])

y=np.array(y)
i=y.size
X=np.array(X)
X=X.reshape(i,1024)
np.save("X_12.npy",X)
np.save("Y_12.npy",y)

'''
        
    #temp = open(d,'r')
    #for line in temp:
    #    y = line[-2:]
    #    print y
    #    if y[0]=="  ":
    #        print y[1]
    f.close()  
         
y=np.array(y)        
    #temp.close()
#y = np.delete(y,0,0)
print ("printing y",y)
print(y.size)

np.save("Y_12.npy",y)
i=y.size


for img in images:
    imgpath = "train_images2/"+img
    temp = cv2.imread(imgpath,0)
    height, width = temp.shape[:2]
    print(height,width)
    #temp = temp.reshape(1,height*width)
    height, width = temp.shape[:2]
    print(height,width)
    
    X.append([np.array(temp)])

X=np.array(X)
X=X.reshape(i,1024)
cv2.imshow("X",X)
cv2.waitKey(0)
print(X.shape)
np.save("X_12.npy",X) 
'''
        
    
        
     
