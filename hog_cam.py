import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
from operator import itemgetter
from datetime import datetime

def addRowsCols(matrice,width,height) :
    col=[[row[0]] for row in matrice]
    matrice = np.append(col, matrice, axis=1)
    width+=1
    col=[[row[width-1]] for row in matrice]
    matrice = np.append(matrice,col, axis=1)
    width+=1

    ligne=[val for val in matrice[0]]
    matrice = np.append([ligne],matrice, axis=0)
    height+=1
    ligne=[val for val in matrice[height-1]]
    matrice = np.append(matrice,[ligne], axis=0)
    height+=1
    return matrice,width,height

def direction(image,w,h):

    gx = np.zeros([h,w])
    for p in range(1,w+1) :
        for l in range(1,h+1) :
            gx[l-1,p-1] = int(image[l+1,p]) - int(image[l,p])
    
    gy = np.zeros([h,w])
    for l in range(1,h+1) :
        for p in range(1,w+1) :
            gy[l-1,p-1] = int(image[l,p+1]) - int(image[l,p])

    m = np.zeros([h,w])
    for l in range(h) :
        for p in range(w) :
            m[l,p]= math.sqrt( gx[l,p]**2 + gy[l,p]**2 )



    d = np.zeros([h,w])
    for l in range(h) :
        for p in range(w) :
            d[l,p]= math.degrees(np.arctan2( gy[l,p] , gx[l,p]))
            d[l,p]=math.floor(d[l,p])%360
    return gx,gy,m,d

#fonction pour retouner un histogramme de taille n : classes     
def HOG_8x8(direction,magnitude , yi,xj,n):
    r=360/n
    hist_x=np.arange(0,360,r)
    hist_y=np.zeros(n)
    for i in range(yi,yi+8):
        for j in range(xj,xj+8):
            d=int(direction[i,j]/r)
            
            mo=direction[i,j]%r
            if(mo==0):
               hist_y[d]+=magnitude[i,j]   
            else :    
               hist_y[d]+=magnitude[i,j]*mo/r
               d=(d+1)%n
               hist_y[d]+=(magnitude[i,j]*(1-mo/r))
    # hist_x=np.arange(0,36)           
    # plt.bar(hist_x, hist_y, color ='yellow',width = r,edgecolor='red',align='edge',hatch="/")
    # plt.xlabel("Orientation bins : "+ str(n) +" classes")
    # plt.ylabel("N° pixels per class")
    # plt.xticks(np.arange(0,361,r))
    # plt.title("HOG")
    # plt.show()
  
    return hist_y
#fonction pour une matrice des histogrames de taille (128/8 , 128/8 , n )
#donc chaque element de la matrice est un vecteur de taille n     
def HogHists_8x8(direction,magnitude , width,height,n):
    hists=np.array([])
    count=0
    for i in range(0,height,8):
        for j in range(0,width,8) :  
            hist_y=HOG_8x8(direction,magnitude,i,j,n)
            hists=np.append(hists,hist_y,axis=0) 
    hists=hists.reshape(int(height/8),int(width/8),n)
    return hists        

#une fonction pour calculer de descripteur a partir de la matrice precedente
#pour chaque 4 blocks 8*8 de la matrice  on construit un descripteur de taille n*4     
# 1 2 3 
# 4 5 6
# 7 8 9
def Descripteur(dir,mag,width,height,nb):
    hists=HogHists_8x8(dir,mag,width,height,nb)
    h,w,_=hists.shape
    desc=np.array([])
    for i in range (h-1):
        for j in range(w-1):
            hist_2x2=np.array([])
            hist_2x2=np.append(hist_2x2,hists[i][j])
            hist_2x2=np.append(hist_2x2,hists[i][j+1])
            hist_2x2=np.append(hist_2x2,hists[i+1][j])
            hist_2x2=np.append(hist_2x2,hists[i+1][j+1])
            m=0
            for k in hist_2x2:
                m+=k**2

            m=math.sqrt(m)       

            for ki in range(len(hist_2x2)):
                hist_2x2[ki]/=m 

            desc=np.append(desc,hist_2x2)
            #hist_x=np.arange(0,36)
            # plt.bar(hist_x, hist_2x2, color ='yellow',width = 1,edgecolor='red',align='edge',hatch="/")
            # plt.xlabel("Orientation bins : "+ str(n) +" classes")
            # plt.ylabel("N° pixels per class")
            # plt.xticks(np.arange(0,37))
            # plt.title("HOG")
            # plt.show()      
    return desc        


def Sim(desc1,desc2):
    mse=0;
    for i in range(np.size(desc1)):
        mse=mse+(desc1[i]-desc2[i])**2 
    return mse/len(desc1)


def comparerFace(imagec,image0,folderpath,nbb):
    compareMse=[]
    
    #step1: trouver la face dans l'image initiale et calculer descripteur hog  
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image0, 1.1, 4)
    if(len(faces)==0):
        return []
    for (x, y, w, h) in faces:
        image1=image0[y:h+y,x:w+x]
        cv2.rectangle(imagec, (x, y), (x + w, y + h), (255, 0, 0), 2)  
    plt.figure("Face Detection")
    plt.imshow(np.array(imagec))
    plt.axis("off")    

    width=128
    height=128



    image1 = cv2.resize(image1, (width,height))
    matrice1 = np.asarray(image1)
    matrice1,_,_=addRowsCols(matrice1,width,height)        
    _,_,mag1,dir= direction(matrice1,width,height)    
    desc1=Descripteur(dir,mag1,width,height,nbb)
    

    #step2:lister les images a comparer avec l'image initiale 
    # et trouver la face pour chacune des images
    list_files=[]
    for path in os.listdir(folderpath):
        full_path = os.path.join(folderpath, path)
        if os.path.isfile(full_path):
            list_files.append(full_path)


    list_faces=[]
    width=128
    height=128
    for j in range(0,len(list_files)):
        image01=cv2.imread(list_files[j],0)
        faces2 = face_cascade.detectMultiScale(image01, 1.1, 4)
        for (x, y, w, h) in faces2:
            image=image01[y:h+y,x:w+x]
            image = cv2.resize(image, (width,height))
            list_faces.append(np.asarray(image)) 
    


    fig=plt.figure("HOG Test",figsize=(10,7))
    n=len(list_faces)+1
    fig.add_subplot(1,n,1)
    plt.imshow(image1,cmap="gray")
    plt.axis("off")
    
    #calculer descrpiteur hog pour chaque face dans l'image et affcher les resulats 
    #ajouter le resulats de la comparaison dans une liste
    for i in range(n-1):
        matrice2 = list_faces[i]
        width=128
        height=128
        matrice2,_,_=addRowsCols(matrice2,width,height)           
        _,_,mag2,dir2= direction(matrice2,width,height)    
        desc2=Descripteur(dir2,mag2,width,height,nbb)          
        fig.add_subplot(1,n,i+2)
        plt.imshow(list_faces[i],cmap="gray")
        mse=round(Sim(desc1,desc2),2)
        if(i>=len(list_files)):
            list_files.append(list_files[i-1])
        k={'name':list_files[i]}
        k['MSE']=mse
        compareMse.append(k)
        plt.title(str(mse))

        plt.axis("off")
    
    #trier las liste
    compareMse = sorted(compareMse, key=itemgetter('MSE'))
    fig.tight_layout()
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return compareMse


def showcam2(n,folderPath):
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0
    while count < 1:
        (_, image) = webcam.read()
        cv2.imwrite(folderPath+"ME__"+datetime.now().strftime("%d %m %Y %H %M %S")+".png",np.asarray(image))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        list_mse=comparerFace(image,gray,folderPath,n)  
        for i in range(len(list_mse)):   
            print(list_mse[i])
        count += 1
    webcam.release()
    cv2.destroyAllWindows()








showcam2(9,"test2/")