#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:09:11 2022

inspired by https://lastresortsoftware.blogspot.com/2012/09/fitting-ellipse-to-point-data.html

@author: basile
https://github.com/basilegraf/ellipse-fitting
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import scipy as sc
import scipy.linalg as sla
import control as ctl

# Fit ellipse by fitting the coefficients of the equation 
#     a * x^2 + b * y^2 + c * x*y + d * x + e * y + f = 0
# to the data points
# returns coeffs =  [a,b,c,d,e,f]
def fitEllipse(pts):
    pts = np.asarray(pts)
    n = pts.shape[1]
    x = pts[0,:]
    y = pts[1,:]
    x2 = x * x
    y2 = y * y
    xy = x * y
    cst = np.ones((n))
    A = np.asarray([x2, y2, xy, x, y, cst]).transpose()
    AA = A.transpose() @ A
    w, v = np.linalg.eigh(AA)
    coeffs = v[:,0] # eigen vector to smallest eigen value
    return coeffs
    
# Transforms the ellipse implicit equation 
#     a * x^2 + b * y^2 + c * x*y + d * x + e * y + f = 0
# into ellipse parameters 
# [x0, y0] : ellipse center
# theta : rotation of ellipse axes
# semi1, semi2 : length of semi axes (not ordered)
def transformEllipse(coeffs):
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    d = coeffs[3]
    e = coeffs[4]
    f = coeffs[5]
    discr = 4 * a * b - c**2
    x0 = -(2 * b * d - c * e) / discr
    y0 = -(2 * a * e - c * d) / discr
    theta = 0.5 * np.arctan2(c, a - b)
    aa = 0.5 * (a + b + (a - b) * np.cos(2 * theta) + c * np.sin(2 * theta))
    bb = 0.5 * (a + b + (b - a) * np.cos(2 * theta) - c * np.sin(2 * theta))
    ff = f + d * x0 + a * x0**2 + y0 * (e + c * x0 + b * y0)
    semi1 = np.sqrt(np.abs(ff / aa))
    semi2 = np.sqrt(np.abs(ff / bb))
    return (x0, y0, theta, semi1, semi2)
    

# Build ellipse points from parameters
def ellipsePoints(x0, y0, theta, semi1, semi2, alpha):
    alpha = np.asarray(alpha)
    ax1 = semi1 * np.asarray([np.cos(theta), np.sin(theta)])
    ax2 = semi2 * np.asarray([-np.sin(theta), np.cos(theta)])
    n = len(alpha)
    pts = np.zeros((2, n))
    for k in range(n):
        pts[:,k] = np.asarray([x0, y0]) + np.cos(alpha[k]) * ax1 + np.sin(alpha[k]) * ax2
    return pts

alpha = np.linspace(0,2 * np.pi, 100)
pts = ellipsePoints(10.5,-0.3,-np.pi/5.3,1.5,5.0, alpha)
pts += 0.3 * np.random.normal(size=(2,100))
coeffs = fitEllipse(pts)
params = transformEllipse(coeffs)
pts2 = ellipsePoints(*params, alpha)


plt.axes().set_aspect('equal')
plt.plot(pts[0,:], pts[1,:])
plt.plot(pts2[0,:], pts2[1,:], '--')
plt.grid(True)
plt.title('Many noisy points')



alphaRand = np.pi * np.random.normal(size=(5))
pts = ellipsePoints(10.5,-0.3,-np.pi/5.3,1.5,5.0, alphaRand)
coeffs = fitEllipse(pts)
params = transformEllipse(coeffs)
pts2 = ellipsePoints(*params, alpha)

plt.figure()
plt.axes().set_aspect('equal')
plt.plot(pts[0,:], pts[1,:],'*')
plt.plot(pts2[0,:], pts2[1,:], '--')
plt.grid(True)
plt.title('5 random points part of an ellipse')
