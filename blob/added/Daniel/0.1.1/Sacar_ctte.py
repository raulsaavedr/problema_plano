# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:26:41 2020

@author: Ignacio Useche
"""
import csv

def main():
    lista = []
    csv_final=[]
    with open('csv/salida/matriz_coeficientes.csv') as file:
        csv_reader =csv.reader(file)
        
        for i in csv_reader:
            lista.append(i)
    
    print(f'numero de constantes: {len(lista[1])-1}')
    print(f'dimension de cada constante: {len(lista)-1}')
    
    for numero_cttes in range(1,len(lista[1])):
        csv_final.append([])
        
        for n_dimension in range(1,len(lista)):
            csv_final[numero_cttes-1].append(lista[n_dimension][numero_cttes])    
    
    with open('csv/salida/coeficientes_for_matlab.csv','w') as file:
        writer = csv.writer(file,delimiter=';')
        writer.writerows(csv_final)
        
if __name__=="__main__":
    main()