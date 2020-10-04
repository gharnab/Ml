# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:32:00 2020

@author: shamaun
"""

import numpy as np

arr = np.array([1,2,3])

arr2d = np.array([[1,2,3],
                  [4,5,6]])

arr1 = np.array([[1,2,3],
                 [4,5,6],
                 [7,8,9]])
arr2 = np.array([[10,11,12],
                 [13,14,15],
                 [16,17,18]])

arr3d = np.dstack((arr1,arr2))

#creating special arrays..

zero = np.zeros((3,4))

zeros_like = np.zeros_like(arr3d)

ones = np.ones((4,3))

ones_like = np.ones_like(arr3d)

full = np.full((4,3),'a')

full_like = np.full_like(arr3d,255)

identity = np.eye(4)

#selecting data within an interval

n_pts = 5

lin = np.linspace(0,100,n_pts)

arng = np.arange(0,10,3)

#random data distribution

rand = np.random.rand(3,3)

rand2 = np.random.rand(3,3)*20

randint = np.random.randint(10,200,(3,4))

uni = np.random.uniform(10,20,(3,4))

nrml = np.random.normal(10,2,10)

#indexing and slicing of array

arr = np.random.randint(0,50,(5,5))

arr[3,3]

"""
             ROW                 COL
var_name[start:stop+1:step , start:stop+1:step]

"""
arr[0:5:2 , 0:5:2]

#Operating on arrays

#Scalar operations

arr = np.array([[1,2,3],
                [4,5,6],
                [7,8,9]])

add = np.add(arr,10)
sub = np.subtract(arr, 2)
mul = np.multiply(arr, 2)
div = np.divide(arr,3)
exp = np.power(arr,2)

#vector operations

arr2 = np.array([[11,12,13],
                 [14,15,16],
                 [17,18,19]])

vadd = np.add(arr,arr2)
vsub = np.subtract(arr, arr2)
vmul = np.multiply(arr, arr2) #element wise multiplication
vdiv = np.divide(arr,arr2)
vexp = np.power(np.int64(arr),np.int64(arr2))

#Matrix Multiplication

mat_mul = np.matmul(arr,arr2)

"""_________________________________________"""









