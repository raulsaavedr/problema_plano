# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:31:25 2020

@author: Ignacio Useche
"""
import csv

letras = [chr(char) for char in range(97,97+18) ]
matriz_coeficientes=[]
for letra in letras:
    matriz_coeficientes.append([letras[13].upper()+letra+str(i+1)+'=Cero' for i in range(len(letras))])
    
para_leer=[\
        [1,2,3],\
        [2,3,4,5],\
        [1,2,3],\
        [2,3,4,5,7,8,9],\
        [4,5,10,11],\
        [4,5,6,10,11],\
        [2,3,7,8],\
        [7,8,12,13],\
        [2,3,9],\
        [6,10,11],\
        [10,11,14,15],\
        [7,8,12,13],\
        [12,13,16],\
        [14,15,18],\
        [10,11,14,15],\
        [12,13,16],\
        [14,15,17],\
        [14,15,17,18],\
        ]

for i,fila in enumerate(matriz_coeficientes):
    for x in para_leer[i]:
        aux = fila[x-1].split('C')[0]
        fila[x-1]= aux +input(aux)
        print(fila[x-1])


#ESTO PIDE CADA VALOR
#for fila in matriz_coeficientes:
#    for i, item in enumerate(fila):
#        cad = input(item+' ')
#        if cad =='': 
#            cad='Cero'
#        fila[i] += cad
#        print(fila[i])


with open('matriz_coeficientes.csv','w') as file:
    writer = csv.writer(file)
    writer.writerows(matriz_coeficientes)
