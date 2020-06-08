import csv
def pintar(matriz,vector_de_ecuaciones):
    cuenta = 0
    print()
    print('****************************************')
    for row in range(len(matriz)):

        if vector_de_ecuaciones[row]>9:
            espacios = ' '
        else:
            espacios = '  '

        print(f'{vector_de_ecuaciones[row]}{espacios}',end=': ')
        for col in range(len(matriz)):
        #print(f'{v[i]}')
            if row == col:
                if matriz[row][col]:
                    print('*',end=' ')
                    cuenta += 1
                else:
                    print('-',end=' ')
            else:
                print(matriz[row][col],end=' ')
        print()

    print('****************************************')
    print(f'{cuenta} de {len(matriz)} 1\'s en la diagonal')

def ordenar(matriz,n,v):
    columna_seleccionada = []
    filas_candidatas = []
    m = n-1
    min_count = 1e100
    fila_min = m
    #print(f'iteracion: {len(matriz)-n}')
    for row in range(m):
        if matriz[row][m] == 1:
            filas_candidatas.append(row) #selecciona filas con valor 1 de la columna m

    if not verificar(matriz): #verifica si la diagonal esta completa

        if len(filas_candidatas):   #escoje las filas apropiadas para poner en la posicion m
            for fila in filas_candidatas:
                cuenta = 0
                for columna in range(m-1):
                    cuenta += matriz[fila][columna]*(len(matriz)-columna) #califica las columnas como un sistema posicional
                if cuenta < min_count: # las filas con menores ocurrencias son mas probables a ser tomadas
                    min_count = cuenta
                    fila_min = fila
                    
            matriz[fila_min],matriz[m] = matriz[m],matriz[fila_min]
            v[fila_min],v[m] = v[m],v[fila_min]

        else: # si no existen filas candidatas
            m = len(matriz)
            print('''

            !!!reiniciar¡¡¡

            ''')

        ordenar(matriz,m,v)

    else: #si la diagonal ya esta completa
        return

def leerMatriz(ruta):
    constantes = []
    with open(ruta) as file:
        reader = csv.reader(file)
        for C in reader:
            constantes.append(C)
            
    maxima_const = 0
    for list_constante in constantes:
        for constante in list_constante:
            if int(constante.split('C')[1]) > maxima_const:
                maxima_const = int(constante.split('C')[1])
    constantes_int = [[0 for i in range(1, maxima_const + 1)] for j in range(1, maxima_const + 1)]
    # Se crea una lista de listas con unos donde estan C1,C2,CN y ceros donde no hay valores
    j = 0
    for list_constante in constantes:
        for constante in list_constante:
            constantes_int[j][int(constante.split('C')[1]) - 1] = 1
        j += 1
    
    return constantes_int   

def verificar(matriz):
    cuenta = 0
    for i in range(len(matriz)):
        if matriz[i][i]:
            cuenta += 1
    if cuenta == len(matriz):
        return True
    else:
        return False

if __name__ == '__main__':
    matriz_de_prueba = leerMatriz('csv/constantes_c.csv')
    vect_filas = list(range(1,len(matriz_de_prueba)+1))

    print('''
            MATRIZ ORIGINAL''')
    pintar(matriz_de_prueba,vect_filas)

    ordenar(matriz_de_prueba,len(matriz_de_prueba),vect_filas)

    print('''
            MATRIZ ORDENADA''')
    pintar(matriz_de_prueba,vect_filas)
