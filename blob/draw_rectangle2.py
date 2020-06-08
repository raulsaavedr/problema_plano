# -*- coding: utf-8 -*-
"""
Created on Wed May 20 21:05:48 2020

@author: Ignacio Useche
"""

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
import mpl_toolkits.mplot3d.art3d as art3d

def draw_rectangle(ax= None, inicio= 0, ancho= 2,direction = 'y',desp= 2, alto= 3, fill= True):
    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    rect = Rectangle((inicio,0), width= ancho, height=alto, fill= fill)
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=desp, zdir=direction)
    

if __name__=='__main__':

    z  = 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #Region 1
    xi, xf = (2,4)
    yi, yf = (0,1)
    draw_rectangle(ax, inicio= xi, ancho= xf-xi, direction= 'y', desp= yi, alto= z)
    draw_rectangle(ax, inicio= xi, ancho= xf-xi, direction= 'y', desp= yf, alto= z, fill= False)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xi, alto= z)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xf, alto= z)
    
    #region 2
    xi, xf = (0,6)
    yi, yf = (1,2)
    draw_rectangle(ax, inicio= 0, ancho= 2, direction= 'y', desp= yi, alto= z)
    draw_rectangle(ax, inicio= 4, ancho= 2, direction= 'y', desp= yi, alto= z)
    draw_rectangle(ax, inicio= xi, ancho= xf-xi, direction= 'y', desp= yf, alto= z, fill= False)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xi, alto= z)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xf, alto= z, fill= False)
    
    #region 3
    xi, xf = (0,6)
    yi, yf = (2,3)
    draw_rectangle(ax, inicio= xi, ancho= xf-xi, direction= 'y', desp= yi, alto= z, fill= False)
    draw_rectangle(ax, inicio= 0, ancho= 2, direction= 'y', desp= yf, alto= z)
    draw_rectangle(ax, inicio= 4, ancho= 2, direction= 'y', desp= yf, alto= z)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xi, alto= z, fill= False)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xf, alto= z)
    
    #region 4
    xi, xf = (2,4)
    yi, yf = (3,4)
    draw_rectangle(ax, inicio= xi, ancho= xf-xi, direction= 'y', desp= yi, alto= z, fill= False)
    draw_rectangle(ax, inicio= xi, ancho= xf-xi, direction= 'y', desp= yf, alto= z, fill= False)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xi, alto= z)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xf, alto= z)
    
    #region 5
    xi, xf = (0,6)
    yi, yf = (4,5)
    draw_rectangle(ax, inicio= 0, ancho= 2, direction= 'y', desp= yi, alto= z)
    draw_rectangle(ax, inicio= 4, ancho= 2, direction= 'y', desp= yi, alto= z)
    #draw_rectangle(ax, inicio= xi, ancho= xf-xi, direction= 'y', desp= yf, alto= z, fill= False)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xf, alto= z, fill= False)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xi, alto= z)
    
    #region 6
    xi, xf = (0,1)
    yi, yf = (5,6)
    draw_rectangle(ax, inicio= xi, ancho= xf-xi, direction= 'y', desp= yi, alto= z)
    draw_rectangle(ax, inicio= xi, ancho= xf-xi, direction= 'y', desp= yf, alto= z, fill= False)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xi, alto= z, fill= False)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xf, alto= z)
    
    #region 7
    xi, xf = (1,2)
    yi, yf = (5,6)
    draw_rectangle(ax, inicio= xi, ancho= xf-xi, direction= 'y', desp= yi, alto= z, fill= False)
    draw_rectangle(ax, inicio= xi, ancho= xf-xi, direction= 'y', desp= yf, alto= z)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xi, alto= z)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xf, alto= z)
    
    #region 8
    xi, xf = (2,3)
    yi, yf = (5,6)
    draw_rectangle(ax, inicio= xi, ancho= xf-xi, direction= 'y', desp= yi, alto= z)
    draw_rectangle(ax, inicio= xi, ancho= xf-xi, direction= 'y', desp= yf, alto= z, fill= False)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xi, alto= z)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xf, alto= z)
    
    #region 9
    xi, xf = (3,5)
    yi, yf = (5,6)
    draw_rectangle(ax, inicio= 5, ancho= 1, direction= 'y', desp= yi, alto= z)
    draw_rectangle(ax, inicio= xi, ancho= xf-xi, direction= 'y', desp= yi, alto= z, fill= False)
    draw_rectangle(ax, inicio= xi, ancho= xf-xi, direction= 'y', desp= yf, alto= z, fill= False)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xi, alto= z)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xf, alto= z)
    
    #region 10
    xi, xf = (0,6)
    yi, yf = (6,7)
    draw_rectangle(ax, inicio= 5, ancho= 1, direction= 'y', desp= yi, alto= z)
    draw_rectangle(ax, inicio= xi, ancho= xf-xi, direction= 'y', desp= yf, alto= z)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xi, alto= z, fill= False)
    draw_rectangle(ax, inicio= yi, ancho= yf-yi, direction= 'x', desp= xf, alto= z, fill= False)
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 7)
    ax.set_zlim(0, 5)