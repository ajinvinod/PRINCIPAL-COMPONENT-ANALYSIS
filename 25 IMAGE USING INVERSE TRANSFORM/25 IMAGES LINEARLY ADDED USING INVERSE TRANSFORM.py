import numpy as np
import matplotlib.pyplot as plt
import matplotlib as plot
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.decomposition import DictionaryLearning as DL

plt.figure(1)

imgA=np.array([[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,0,0,1]])
imgB=np.array([[1,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,1]])
imgC=np.array([[1,1,1,1],[1,0,0,0],[1,0,0,0],[1,1,1,1]])
imgD=np.array([[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,1]])
imgE=np.array([[1,1,1,1],[1,1,0,0],[1,0,0,0],[1,1,1,1]])
imgF=np.array([[1,1,1,1],[1,0,0,0],[1,1,1,0],[1,0,0,0]])
imgG=np.array([[0,1,1,0],[1,0,0,0],[1,0,1,1],[0,1,1,0]])
imgH=np.array([[1,0,0,1],[1,1,1,1],[1,1,1,1],[1,0,0,1]])
imgI=np.array([[1,1,1,1],[0,1,1,0],[0,1,1,0],[1,1,1,1]])
imgJ=np.array([[1,1,1,1],[0,0,0,1],[1,0,0,1],[0,1,1,0]])
imgK=np.array([[1,0,0,1],[1,1,1,0],[1,1,1,0],[1,0,1,1]])
imgL=np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]])
imgM=np.array([[1,0,1,1],[1,1,1,1],[1,1,0,1],[1,0,0,1]])
imgN=np.array([[1,0,0,1],[1,1,0,1],[1,0,1,1],[1,0,0,1]])
imgO=np.array([[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]])
imgP=np.array([[1,1,1,1],[1,0,0,1],[1,1,1,1],[1,0,0,0]])
imgQ=np.array([[1,1,1,1],[1,0,0,1],[1,0,1,1],[1,1,1,1]])
imgR=np.array([[1,1,1,1],[1,0,0,1],[1,1,1,0],[1,0,1,1]])
imgS=np.array([[1,1,1,1],[1,1,0,0],[0,0,1,1],[1,1,1,1]])
imgT=np.array([[1,1,1,1],[0,1,1,0],[0,1,1,0],[0,1,1,0]])
imgU=np.array([[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]])
imgV=np.array([[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]])
imgW=np.array([[1,0,0,1],[1,0,1,1],[1,1,1,1],[1,1,0,1]])
imgX=np.array([[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]])
imgY=np.array([[1,0,0,1],[1,1,1,1],[0,1,1,0],[0,1,1,0]])
#imgZ=np.array([[1,1,1,1],[0,0,1,1],[1,1,0,0],[1,1,1,1]])




plt.subplot(6,5,1),plt.imshow(imgA,cmap='Greys')
plt.subplot(6,5,2),plt.imshow(imgB,cmap='Greys')
plt.subplot(6,5,3),plt.imshow(imgC,cmap='Greys')
plt.subplot(6,5,4),plt.imshow(imgD,cmap='Greys')
plt.subplot(6,5,5),plt.imshow(imgE,cmap='Greys')

plt.subplot(6,5,6),plt.imshow(imgF,cmap='Greys')
plt.subplot(6,5,7),plt.imshow(imgG,cmap='Greys')
plt.subplot(6,5,8),plt.imshow(imgH,cmap='Greys')
plt.subplot(6,5,9),plt.imshow(imgI,cmap='Greys')
plt.subplot(6,5,10),plt.imshow(imgJ,cmap='Greys')

plt.subplot(6,5,11),plt.imshow(imgK,cmap='Greys')
plt.subplot(6,5,12),plt.imshow(imgL,cmap='Greys')
plt.subplot(6,5,13),plt.imshow(imgM,cmap='Greys')
plt.subplot(6,5,14),plt.imshow(imgN,cmap='Greys')
plt.subplot(6,5,15),plt.imshow(imgO,cmap='Greys')

plt.subplot(6,5,16),plt.imshow(imgP,cmap='Greys')
plt.subplot(6,5,17),plt.imshow(imgQ,cmap='Greys')
plt.subplot(6,5,18),plt.imshow(imgR,cmap='Greys')
plt.subplot(6,5,19),plt.imshow(imgS,cmap='Greys')
plt.subplot(6,5,20),plt.imshow(imgT,cmap='Greys')

plt.subplot(6,5,21),plt.imshow(imgU,cmap='Greys')
plt.subplot(6,5,22),plt.imshow(imgV,cmap='Greys')
plt.subplot(6,5,23),plt.imshow(imgW,cmap='Greys')
plt.subplot(6,5,24),plt.imshow(imgX,cmap='Greys')
plt.subplot(6,5,25),plt.imshow(imgY,cmap='Greys')
#plt.subplot(6,5,26),plt.imshow(imgZ,cmap='Greys')

#=====================================================
imgA=imgA+imgB
imgB=imgB+imgC
imgC=imgC+imgD
imgD=imgD+imgE
imgE=imgE+imgF
imgF=imgF+imgG
imgG=imgG+imgH
imgH=imgH+imgI
imgI=imgI+imgJ
imgJ=imgJ+imgK
imgK=imgK+imgL
imgL=imgL+imgM
imgM=imgM+imgN
imgN=imgN+imgO
imgO=imgO+imgP
imgP=imgP+imgQ
imgQ=imgQ+imgR
imgR=imgR+imgS
imgS=imgS+imgT
imgT=imgT+imgU
imgU=imgU+imgV
imgV=imgV+imgW
imgW=imgW+imgX
imgX=imgX+imgY
imgY=imgY+imgA

matrix=np.zeros(shape=[16,25000])
def noise_gen(initial,final,img):
    for i in range(initial,final):
        noise = np. random. normal(0, .1, img.shape)
        img1=img+noise
        img2=img1.ravel()
        matrix[:,i]=img2
    return(0)
imgA1=noise_gen(0,1000, imgA)
imgB1=noise_gen(1000,2000, imgB)
imgC1=noise_gen(2000,3000, imgC)
imgD1=noise_gen(3000,4000, imgD)
imgE1=noise_gen(4000,5000, imgE)
imgF1=noise_gen(5000,6000, imgF)
imgG1=noise_gen(6000,7000, imgG)
imgH1=noise_gen(7000,8000, imgH)
imgI1=noise_gen(8000,9000, imgI)
imgJ1=noise_gen(9000,10000, imgJ)
imgK1=noise_gen(10000,11000, imgK)
imgL1=noise_gen(11000,12000, imgL)
imgM1=noise_gen(12000,13000, imgM)
imgN1=noise_gen(13000,14000, imgN)
imgO1=noise_gen(14000,15000, imgO)
imgP1=noise_gen(15000,16000, imgP)
imgQ1=noise_gen(16000,17000, imgQ)
imgR1=noise_gen(17000,18000, imgR)
imgS1=noise_gen(18000,19000, imgS)
imgT1=noise_gen(19000,20000, imgT)
imgU1=noise_gen(20000,21000, imgU)
imgV1=noise_gen(21000,22000, imgV)
imgW1=noise_gen(22000,23000, imgW)
imgX1=noise_gen(23000,24000, imgX)
imgY1=noise_gen(24000,25000, imgY)
#=========================================
plt.figure(2)
matrix=matrix.T
pca=PCA(n_components=16)
pca.fit(matrix)
pca_trans=pca.fit_transform(matrix)
#pca_comp=pca.components_
pca_comp=pca.inverse_transform(pca_trans)
#inv=pca.inverse_transform(pca_trans)
#trans=pca.transform(matrix)
print(pca_comp.shape)
print(pca_trans.shape)
plt.tight_layout()
plt.subplot(6,5,1)
x0=pca_comp[0].reshape(4,4)
plt.imshow(x0,cmap='Greys')
plt.subplot(6,5,2)
x1=pca_comp[1000].reshape(4,4)
plt.imshow(x1,cmap='Greys')
plt.subplot(6,5,3)
x2=pca_comp[2000].reshape(4,4)
plt.imshow(x2,cmap='Greys')
plt.subplot(6,5,4)
x3=pca_comp[3000].reshape(4,4)
plt.imshow(x3,cmap='Greys')
plt.subplot(6,5,5)
x4=pca_comp[4000].reshape(4,4)
plt.imshow(x4,cmap='Greys')
plt.subplot(6,5,6)
x5=pca_comp[5000].reshape(4,4)
plt.imshow(x5,cmap='Greys')
plt.subplot(6,5,7)
x6=pca_comp[6000].reshape(4,4)
plt.imshow(x6,cmap='Greys')
plt.subplot(6,5,8)
x7=pca_comp[7000].reshape(4,4)
plt.imshow(x7,cmap='Greys')
plt.subplot(6,5,9)
x8=pca_comp[8000].reshape(4,4)
plt.imshow(x8,cmap='Greys')
plt.subplot(6,5,10)
x9=pca_comp[9000].reshape(4,4)
plt.imshow(x9,cmap='Greys')
plt.subplot(6,5,11)
x10=pca_comp[10000].reshape(4,4)
plt.imshow(x10,cmap='Greys')
plt.subplot(6,5,12)
x11=pca_comp[11000].reshape(4,4)
plt.imshow(x11,cmap='Greys')
plt.subplot(6,5,13)
x12=pca_comp[12000].reshape(4,4)
plt.imshow(x12,cmap='Greys')
plt.subplot(6,5,14)
x13=pca_comp[13000].reshape(4,4)
plt.imshow(x13,cmap='Greys')
plt.subplot(6,5,15)
x14=pca_comp[14000].reshape(4,4)
plt.imshow(x14,cmap='Greys')
plt.subplot(6,5,16)
x15=pca_comp[15000].reshape(4,4)
plt.imshow(x15,cmap='Greys')



plt.figure(3)
nmf=NMF(n_components=16)
nmf.fit_transform(abs(matrix))
nmf_trans=nmf.fit_transform(abs(matrix))
#nmf_comp=nmf.components_
nmf_comp=nmf.inverse_transform(nmf_trans)
print(nmf_comp.shape)
print(nmf_trans.shape)
plt.tight_layout()

y0=nmf_comp[0].reshape(4,4)
plt.subplot(6,5,1)
plt.imshow(y0,cmap='Greys')
y1=nmf_comp[1000].reshape(4,4)
plt.subplot(6,5,2)
plt.imshow(y1,cmap='Greys')
y2=nmf_comp[2000].reshape(4,4)
plt.subplot(6,5,3)
plt.imshow(y2,cmap='Greys')
y3=nmf_comp[3000].reshape(4,4)
plt.subplot(6,5,4)
plt.imshow(y3,cmap='Greys')
y4=nmf_comp[4000].reshape(4,4)
plt.subplot(6,5,5)
plt.imshow(y4,cmap='Greys')
y5=nmf_comp[5000].reshape(4,4)
plt.subplot(6,5,6)
plt.imshow(y5,cmap='Greys')
y6=nmf_comp[6000].reshape(4,4)
plt.subplot(6,5,7)
plt.imshow(y6,cmap='Greys')
y7=nmf_comp[7000].reshape(4,4)
plt.subplot(6,5,8)
plt.imshow(y7,cmap='Greys')
y8=nmf_comp[8000].reshape(4,4)
plt.subplot(6,5,9)
plt.imshow(y8,cmap='Greys')
y9=nmf_comp[9000].reshape(4,4)
plt.subplot(6,5,10)
plt.imshow(y9,cmap='Greys')
y10=nmf_comp[10000].reshape(4,4)
plt.subplot(6,5,11)
plt.imshow(y10,cmap='Greys')
y11=nmf_comp[11000].reshape(4,4)
plt.subplot(6,5,12)
plt.imshow(y11,cmap='Greys')
y12=nmf_comp[12000].reshape(4,4)
plt.subplot(6,5,13)
plt.imshow(y12,cmap='Greys')
y13=nmf_comp[13000].reshape(4,4)
plt.subplot(6,5,14)
plt.imshow(y13,cmap='Greys')

y14=nmf_comp[14000].reshape(4,4)
plt.subplot(6,5,15)
plt.imshow(y14,cmap='Greys')
y15=nmf_comp[15000].reshape(4,4)
plt.subplot(6,5,16)
plt.imshow(y15,cmap='Greys')


'''
plt.figure(4)
dl=DL(n_components=16)
dl.fit_transform(matrix)
dl_trans=dl.fit_transform(matrix)
dl_comp=dl_trans @ dl.components_
print(dl_comp.shape)
print(dl_trans.shape)

z0=dl_comp[0].reshape(4,4)
plt.subplot(6,5,1)
plt.imshow(z0,cmap='Greys')

z1=dl_comp[1].reshape(4,4)
plt.subplot(6,5,2)
plt.imshow(z1,cmap='Greys')

z2=dl_comp[2].reshape(4,4)
plt.subplot(6,5,3)
plt.imshow(z2,cmap='Greys')

z3=dl_comp[3].reshape(4,4)
plt.subplot(6,5,4)
plt.imshow(z3,cmap='Greys')

z4=dl_comp[4].reshape(4,4)
plt.subplot(6,5,5)
plt.imshow(z4,cmap='Greys')

z5=dl_comp[5].reshape(4,4)
plt.subplot(6,5,6)
plt.imshow(z5,cmap='Greys')

z6=dl_comp[6].reshape(4,4)
plt.subplot(6,5,7)
plt.imshow(z6,cmap='Greys')

z7=dl_comp[7].reshape(4,4)
plt.subplot(6,5,8)
plt.imshow(z7,cmap='Greys')

z8=dl_comp[8].reshape(4,4)
plt.subplot(6,5,9)
plt.imshow(z8,cmap='Greys')

z9=dl_comp[9].reshape(4,4)
plt.subplot(6,5,10)
plt.imshow(z9,cmap='Greys')

z10=dl_comp[10].reshape(4,4)
plt.subplot(6,5,11)
plt.imshow(z10,cmap='Greys')

z11=dl_comp[11].reshape(4,4)
plt.subplot(6,5,12)
plt.imshow(z11,cmap='Greys')

z12=dl_comp[12].reshape(4,4)
plt.subplot(6,5,13)
plt.imshow(z12,cmap='Greys')

z13=dl_comp[13].reshape(4,4)
plt.subplot(6,5,14)
plt.imshow(z13,cmap='Greys')

z14=dl_comp[14].reshape(4,4)
plt.subplot(6,5,15)
plt.imshow(z14,cmap='Greys')

z15=dl_comp[15].reshape(4,4)
plt.subplot(6,5,16)
plt.imshow(z15,cmap='Greys')

'''




























