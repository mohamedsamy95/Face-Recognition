# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 18:08:10 2018

@author: Nada
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import os
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import glob
import scipy
import numpy.linalg as linalg
from tempfile import TemporaryFile
data = np.array([])
values = TemporaryFile()
vectors=TemporaryFile()

for x in range(1,41,1):
    number = str(x)
    filenames = glob.glob('E:\\8th term\\Pattern Recognition\\att_faces\\orl_faces\\s'+ number +'/*.pgm')
    filenames.sort()
    img = [Image.open(fn).convert('L') for fn in filenames]
    images = np.asarray([np.array(im).flatten() for im in img])
    #label= np.full((10, 1), x)
    #person= np.append(images, label, axis =1)
    if (x==1):
        data = np.array(images)
        train_label =np.full((5, 1), x)
    else:
        data = np.append(data,images,axis = 0)
        train_label= np.append(train_label , np.full((5, 1), x) , axis=0)
        
#spliting data into testing and training
train = data[0:][::2] #even
test =  data[1:][::2] # odd

#bonus of splitting
'''train=data[0:70][::]
test=data[70::][::]'''
vectorsfile = np.array([])
SumValues = np.array([])
valuesfile = np.array([])


#PCA
def PCA(train,test,alpha):
   global SumValues
   global vectorsfile
   global valuesfile
   #np.save("values.npy", values)
   #np.save("vectors.npy", vectors)
   '''mean_vector = np.mean(train, axis=0)
   Z = train - np.transpose(mean_vector)
   precovariance=np.matmul(np.transpose(Z),Z)
   covariance=np.divide(precovariance,200)
   print("calculating")
   values,vectors=np.linalg.eig(covariance)
   values_real = np.real(values)
   vectors_real = np.real(vectors)
   print("done calculating")
   np.save("values.npy", values_real)
   np.save("vectors.npy", vectors_real)'''
   if  vectorsfile.size == 0  :
       print("reading files")
       vectorsfile=np.load("vectors.npy")
       valuesfile=np.load("values.npy")
       #print("eig vecs:")
       #print(vectorsfile)
       #print("eig vals")
       #print(valuesfile)
       SumValues=np.sum(valuesfile)
       #print(SumValues)
       #valuesfile=np.sort(valuesfile)[::-1]
       idx = valuesfile.argsort()[::-1]   
       valuesfile = valuesfile[idx]
       vectorsfile = vectorsfile[:,idx]
   ratio=0
   i=0
   for s in range(0,10304,1):
       ratio=ratio+(valuesfile[s]/SumValues)
       if (ratio<alpha):
           i=i+1
       else:
           print ("number of taken eig vals:")
           print(i)
           break
   reduced_basis=vectorsfile[::,:i]
   reduced_basis_t = np.transpose(reduced_basis)
   print("Shape: ")
   print(reduced_basis.shape)
   reduced_dimension_train = np.matmul(reduced_basis_t,np.transpose(train))
   print(reduced_dimension_train.shape)
   reduced_dimension_test = np.matmul(reduced_basis_t,np.transpose(test))

   return np.transpose(reduced_dimension_train),np.transpose(reduced_dimension_test);

#code for plotting the faces
'''for x in range (0,200,1):
      eigen_face= np.reshape(vectorsfile[::,x],(112,92))
      plt.imshow(eigen_face, cmap='gray')
      plt.show()'''
#LDA
def LDA(train,test):
    '''
    persons = np.split(train,40,axis=0)
    #print(persons)
    U = np.mean(train,axis=0)
    i=1
    for person in persons:
        if(i==1):
            means = np.array([np.mean(person,axis=0)])
            i=0
        else:
            means = np.append(means, [np.mean(person,axis=0)] ,axis=0)

    i=1
    for Ui in means:
        if (i==1):
            Sb = 5* np.matmul( np.transpose( [np.subtract(Ui,U)] ) , [np.subtract(Ui,U)] )
           # print(Sb)
            i=0
        else:
            Sb = np.add( Sb , 5* np.matmul( np.transpose( [np.subtract(Ui,U)] ) , [np.subtract(Ui,U)] ) )
        #print(np.subtract(Ui,U))
        #print(np.transpose( np.subtract(Ui,U) ))
   # print(Sb)
    #print(Sb.shape)
    
    i=1
    for person, Ui in zip(persons,means) :
        Zi = person - Ui
       # print(person)
       # print(Ui)
        #Zi=np.subtract(person,Ui)
       
        
        Si = np.matmul(np.transpose(Zi) , Zi)
        if(i==1):
            S = np.array(Si)
        else:
            S = np.add(S,Si)
    S_inv = np.linalg.pinv(S)
    S_inverse_B=np.matmul(S_inv,Sb)
    U, S, V = np.linalg.svd(S_inverse_B)
    eigen_values, eigen_vectors = np.square(S), V.T
   # eigenval,eigenvec = np.linalg.eig(np.matmul(S_inv,Sb))
    np.save("valuesLDA.npy", np.real(eigen_values))
    np.save("vectorsLDA.npy",np.real(eigen_vectors))
   '''
    xxx= np.load("valuesLDA.npy")
    yyy1= np.load("vectorsLDA.npy")
    
    idx = xxx.argsort()[::-1]   
    xxx = xxx[idx]
    yyy = yyy1[:,idx] 
    print("shape el yyy")
    print(yyy.shape)
    reduced_basis_LDA=yyy[::,:39]
    reduced_basis_t_LDA = np.transpose(reduced_basis_LDA)
    print("Shape: ")
    print(reduced_basis_t_LDA.shape)
    reduced_dimension_train = np.matmul(reduced_basis_t_LDA,np.transpose(train))
    print(reduced_dimension_train.shape)
    reduced_dimension_test = np.matmul(reduced_basis_t_LDA,np.transpose(test))
    '''for x in range (0,2000,1):
       eigen_face= np.reshape(yyy[x,::],(112,92))
       plt.imshow(eigen_face, cmap='gray')
       plt.show()'''
    return np.transpose(reduced_dimension_train),np.transpose(reduced_dimension_test);


    


#k-nearest neighnour
def k_nearest(train,test,k):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(train,train_label)
    y_expect = train_label
    y_predict = knn.predict(test)
    print(metrics.classification_report(y_expect,y_predict ))
    
#code done for calculating eigen vals
'''mean_vector = np.mean(train, axis=0)
Z = train - np.transpose(mean_vector)
precovariance=np.matmul(np.transpose(Z),Z)
covariance=np.divide(precovariance,200)
print("calculating")
values,vectors=np.linalg.eig(covariance)
values_real = np.real(values)
vectors_real = np.real(vectors)
print("done calculating")
np.save("values.npy", values_real)
np.save("vectors.npy", vectors_real)'''

'''train1,test1=PCA(train,test,0.8)
train2,test2=PCA(train,test,0.85)
train3,test3=PCA(train,test,0.9)
train4,test4=PCA(train,test,0.95)
train5,test5=PCA(train,test,1)

#k_nearest(train4,test4,1)
#k_nearest(train,test,3)
#k_nearest(train,test,5)
#k_nearest(train,test,7)'''

'''k_nearest(train1,test1,1)
k_nearest(train2,test2,3)
k_nearest(train3,test3,5)
k_nearest(train4,test4,7)
k_nearest(train5,test5,1)'''

train7,test7=LDA(train,test)
k_nearest(train7,test7,1)