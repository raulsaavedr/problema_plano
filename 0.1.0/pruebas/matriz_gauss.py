import numpy as np

def calcular_matriz_gauss_prueba_1():
    # Prueba de una matriz de 5x5
    matriz_coeficientes = np.array([[1, 2, 3, 0, 0],
                                    [0, 3, 4, 5, 0],
                                    [0, 0, 5, 6, 7],
                                    [0, 0, 0, 7, 8],
                                    [0, 0, 1, 0, 4]])
    vectores_valores_dependientes = np.array([[1],
                                              [0],
                                              [5],
                                              [0],
                                              [1]])
    #print(matriz_coeficientes)
    #print(vectores_valores_dependientes)
    m_gauss = np.linalg.solve(matriz_coeficientes, vectores_valores_dependientes)
    print(m_gauss)


def calcular_matriz_gauss_prueba_2():
    # Prueba de una matriz de 5x5
    P = np.array([[1, 2, 3, 0, 0],
                  [0, 3, 4, 5, 0],
                  [0, 0, 5, 6, 7],
                  [0, 0, 0, 7, 8],
                  [0, 0, 1, 0, 4]])
    vectores_valores_dependientes = np.array([[1],
                                              [0],
                                              [5],
                                              [0],
                                              [1]])
    #print(matriz_coeficientes)
    #print(vectores_valores_dependientes)
    m_gauss = np.linalg.solve(matriz_coeficientes, vectores_valores_dependientes)
    print(m_gauss)

def main():
    calcular_matriz_gauss()

if __name__ == '__main__':
    main()
