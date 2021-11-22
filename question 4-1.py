# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 00:10:39 2021

@author: Lenovo
"""

def myHist(mat):
    
    R=input('Enter a vlue')
    C=input('e')
    print(R)
    print(C)
    RR=int(R)
    CC=int(C)
    matrix = []
    print('Enter value raw wise')
    for i in range(RR):
        a=[]
        for j in range(CC):
            a.append(int(input()))
        matrix.append(a)
    print(matrix)
myHist('11')
    