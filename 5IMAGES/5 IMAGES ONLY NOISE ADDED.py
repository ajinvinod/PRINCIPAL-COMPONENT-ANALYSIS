import numpy as np
import matplotlib.pyplot as plt
import matplotlib as plot
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.decomposition import DictionaryLearning as DL

plt.figure()
imgA=np.array([[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,0,0,1]])
imgF=np.array([[1,1,1,1],[1,0,0,0],[1,1,1,0],[1,0,0,0]])
imgL=np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]])
imgJ=np.array([[1,1,1,1],[0,0,1,0],[0,0,1,0],[1,1,1,0]])
imgH=np.array([[1,0,0,1],[1,1,1,1],[1,1,1,1],[1,0,0,1]])
plt.subplot(451)
plt.imshow(imgA,cmap='Greys')
plt.subplot(452)
plt.imshow(imgF,cmap='Greys')
plt.subplot(453)
plt.imshow(imgJ,cmap='Greys')
plt.subplot(454)
plt.imshow(imgH,cmap='Greys')
plt.subplot(455)
plt.imshow(imgL,cmap='Greys')
matrix=np.zeros(shape=[16,5000])
#jA=0
for i in range(0,1000):

    noise = np. random. normal(1, .1, imgA.shape)
    imgA1=imgA+noise
    imgA2=imgA1.ravel()
    matrix[:,i]=imgA2
    #name='plot/pic'+str(jA)
    #plt.imsave(name+'.png',imgA1)
    #jA+=1

for i in range(1000,2000):

    noise = np. random. normal(0, .1, imgF.shape)
    imgF1=imgF+noise
    imgF2=imgF1.ravel()
    matrix[:,i]=imgF2
    #name='plot/pic'+str(jA)
    #plt.imsave(name+'.png',imgA1)
    #jA+=1
for i in range(2000,3000):

    noise = np. random. normal(0, .1, imgJ.shape)
    imgJ1=imgJ+noise
    imgJ2=imgJ1.ravel()
    matrix[:,i]=imgJ2
    #name='plot/pic'+str(jA)
    #plt.imsave(name+'.png',imgA1)
    #jA+=1
for i in range(3000,4000):

    noise = np. random. normal(0, .1, imgH.shape)
    imgH1=imgH+noise
    imgH2=imgH1.ravel()
    matrix[:,i]=imgH2
    #name='plot/pic'+str(jA)
    #plt.imsave(name+'.png',imgA1)
    #jA+=1
for i in range(4000,5000):

    noise = np. random. normal(0, .1, imgL.shape)
    imgL1=imgL+noise
    imgL2=imgL1.ravel()
    matrix[:,i]=imgL2
    #name='plot/pic'+str(jA)
    #plt.imsave(name+'.png',imgA1)
    #jA+=1
matrix=matrix.T

pca=PCA(n_components=5)
pca.fit(matrix)
pca_trans=pca.fit_transform(matrix)
pca_comp=pca.components_
#trans=pca.transform(matrix)
print('pca components shape:',pca_comp.shape)
plt.tight_layout()
plt.subplot(456)
x=pca_comp[0].reshape(4,4)
plt.imshow(x,cmap='Greys')
plt.subplot(457)
x1=pca_comp[1].reshape(4,4)
plt.imshow(x1,cmap='Greys')
plt.subplot(458)
x2=pca_comp[2].reshape(4,4)
plt.imshow(x2,cmap='Greys')
plt.subplot(459)
x3=pca_comp[3].reshape(4,4)
plt.imshow(x3,cmap='Greys')
plt.subplot(4,5,10)
x4=pca_comp[4].reshape(4,4)
plt.imshow(x4,cmap='Greys')

nmf=NMF(n_components=5)
nmf.fit_transform(abs(matrix))
nmf_trans=nmf.fit_transform(abs(matrix))
nmf_comp=nmf.components_
print('nmf component shape:',nmf_comp.shape)

plt.tight_layout()
y1=nmf_comp[0].reshape(4,4)
plt.subplot(4,5,11)
plt.imshow(y1,cmap='Greys')

y2=nmf_comp[1].reshape(4,4)
plt.subplot(4,5,12)
plt.imshow(y2,cmap='Greys')

y3=nmf_comp[2].reshape(4,4)
plt.subplot(4,5,13)
plt.imshow(y3,cmap='Greys')

y4=nmf_comp[3].reshape(4,4)
plt.subplot(4,5,14)
plt.imshow(y4,cmap='Greys')

y5=nmf_comp[4].reshape(4,4)
plt.subplot(4,5,15)
plt.imshow(y5,cmap='Greys')
plt.tight_layout()
dl=DL(n_components=5)
dl.fit_transform(matrix)
dl_trans=dl.fit_transform(matrix)
dl_comp=dl.components_
print(dl_comp.shape)
print(dl_trans.shape)
z1=dl_comp[0].reshape(4,4)
plt.subplot(4,5,16)
plt.imshow(z1,cmap='Greys')

z2=dl_comp[1].reshape(4,4)
plt.subplot(4,5,17)
plt.imshow(z2,cmap='Greys')

z3=dl_comp[2].reshape(4,4)
plt.subplot(4,5,18)
plt.imshow(z3,cmap='Greys')
z4=dl_comp[3].reshape(4,4)
plt.subplot(4,5,19)
plt.imshow(z4,cmap='Greys')
z5=dl_comp[4].reshape(4,4)
plt.subplot(4,5,20)
plt.imshow(z5,cmap='Greys')






