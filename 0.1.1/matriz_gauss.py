import numpy as np
import pandas as pd
import csv

import eigen as eig
import region as reg
import hiperbolica as hyp
import matrices_acomplamiento as m_acop
import distorsionador as v_dist
import constantes_k as const_k
import parser_matriz_gauss as parser

__doc__ = """
Este modulo se determina el sistema de ecuaciones Ax = B
siendo el resultado x las constantes C1,C2,...,Cn, siendo (A) la matriz de coe-
ficientes y (B) la columna de terminos constantes.
Es equivalente f09_Matriz_de_Gauss() y a f10_Determine_Constantes()
"""

def calcular_matriz_gauss(vectores_valores_eigen, matrices_diagonales_valores_eig,
                          matrices_diagonales_hiperbolicas, matrices_acomplamiento,
                          vectores_distorsionadores, matrices_acomplamiento_trans, n_dimension=100):
    """
    Funcion encargada de calcular el resultado del sistema de ecuaciones Ax = B
    siendo el resultado (x) las constantes C1,C2,...,Cn, siendo (A) la matriz de
    coeficientes y (B) la columna de terminos constantes.
    Es equivalente a f10_Determine_Constantes()
    Parametros de entrada:
        * constantes_k: Es un dataframe que almacena el calculo de las constan-
          tes k y la forma en que se puede imprimir (calcular_str)
        * vectores_valores_eigen: Es un DataFrame en donde se almacenan todos
          los valores eigen representativos evaluados de 1 hasta n_dimension.
        * matrices_diagonales_valores_eig: Es un DataFrame que almacena las dia-
          gonales de los vectores eigen.
        * matrices_diagonales_hiperbolicas: Es el DataFrame que almacena el cal-
          culo de las matrices diagonales de las funciones hiperbolicas.
        * matrices_acomplamiento: Es un DataFrame donde se encuentran los calcu-
          los de las matrices de acomplamiento, puede ser soluciones analitica o
          soluciones con integracion.
        * vectores_distorsionadores: Es un DataFrame donde se encuentran los
          calculos de los vectores_distorsionadores, puede ser soluciones anali-
          tica o soluciones con integracion.
        * matrices_acomplamiento_trans: Es un DataFrame creado para facilitar el
          calculo y contiene todas las matrices de acoplamiento transpuestas.
        * n_dimension: Dimension de los vectores y matrices del calculo del pro-
          blema plano.

    Parametros de Salida:
        * matriz_gauss_sol_df: Es un dataframe que contiene el valor de todas
        las constantes calculadas.
    """
    # Se cargan las constantes k utilizadas
    constantes_k = const_k.cargar_constantes_k()
    # A traves del parser se carga la matriz de coeficientes(A)
    matriz_coeficientes_calcular_str = parser.cargar_matriz_coeficientes(n_dimension)
    # A traves del parser se carga los terminos constantes (B)
    vectores_valores_dependientes_calcular_str = parser.cargar_variable_valores_dependientes(n_dimension)
    # Se obtiene la dimension de la matriz de gauss, esta dependera del numero de constantes que se tenga
    n_dimension_matriz_gauss = parser.get_n_dimension_matriz_gauss()
    # Extraemos el index del df esto es A,B,C,D,E,F,...,O
    letras_str = matriz_coeficientes_calcular_str.index
    # Extraemos las columnas del df esto es C1,C2,C3,...,C15
    constantes_str = matriz_coeficientes_calcular_str.columns
    # Creamos el MultiIndex para poder crear correctamente el df de la matriz de gauss
    index = pd.MultiIndex.from_tuples([(letra, i) for letra in matriz_coeficientes_calcular_str.index for i in range(1, n_dimension + 1)])
    columns = pd.MultiIndex.from_tuples([(constante, i) for constante in matriz_coeficientes_calcular_str.columns for i in range(1, n_dimension + 1)])
    matriz_coeficientes = matriz_coeficientes_calcular_str.reindex(index)
    matriz_coeficientes = matriz_coeficientes.reindex(columns, axis=1)
    vectores_valores_dependientes = vectores_valores_dependientes_calcular_str.reindex(index=index)
    # Se evalua cada posicion de la matriz de gauss es decir se evalua la expresion contenida en: Na1, Na2,Na3,...No13,No14,No15
    print(f"\n\tCreando matriz de coeficientes y vectores de terminos \n\t\t\t   independientes\n")
    for letra in letras_str:
        vectores_valores_dependientes.loc[letra] = eval(vectores_valores_dependientes_calcular_str.loc[letra])
        for constante_str in constantes_str:
            #print(letra,constante_str)
            #print(matriz_coeficientes_calcular_str.loc[letra, constante_str])
            matriz_coeficientes.loc[letra, constante_str] = eval(matriz_coeficientes_calcular_str.loc[letra, constante_str])
            #print(matriz_coeficientes.loc[letra, constante_str].to_numpy(), matriz_coeficientes.loc[letra, constante_str].to_numpy().shape)
    # Guardar en csv
    # matriz_coeficientes.to_csv('csv/salida/matriz_coeficientes.csv')
    print("\n\t\t\tMatriz de coeficientes")
    print(matriz_coeficientes)
    # print(matriz_coeficientes.to_numpy())
#B = vectores_valores_dependientes.to_numpy(dtype='float').reshape(n_dimension*n_dimension_matriz_gauss,1)
#print(f"\n\t\tVectores de valores de dependientes numpy B: {B.shape}\n{B}")
    print("\n\t\tVectores de valores de dependientes string solo D")
    print(vectores_valores_dependientes.loc['D'].to_string())
    # print(vectores_valores_dependientes.to_numpy().reshape(n_dimension_matriz_gauss,n_dimension))
    # Se calcula con numpy la solucion
#print("\n\t\tS1")
#print(v_dist.sol_analitica_sin(0, 6, vectores_valores_eigen.loc['Q'].to_numpy(), 0, 0))
    print("\n\t\tCalculando solucion con numpy Ax=B ...")
    matriz_gauss_sol = np.linalg.solve(matriz_coeficientes.to_numpy(dtype='float'), vectores_valores_dependientes.to_numpy(dtype='float').reshape(n_dimension*n_dimension_matriz_gauss,1))
    # print(matriz_gauss_sol)
    columns = ['C' + str(i) for i in range(1, n_dimension_matriz_gauss + 1)]
    index = [i for i in range(1, n_dimension + 1)]
    matriz_gauss_sol_df = pd.DataFrame(matriz_gauss_sol.reshape(n_dimension, n_dimension_matriz_gauss), index=index, columns=columns)
    x = matriz_coeficientes.to_numpy(dtype='float').dot(matriz_gauss_sol)
#print(f"\n\t\tComprobando la solucion x:{x.shape}\n{x}\n{np.allclose(matriz_coeficientes.to_numpy().dot(matriz_gauss_sol),B)}")
#print(f"aNOTHER THING:{np.linalg.cond(matriz_coeficientes.to_numpy(dtype='float'))}")
    return matriz_gauss_sol_df

def main():
    # Se cargan las regiones respectivas
    regiones = reg.cargar_regiones(n_dimension=5)  # region.cargarRegiones(2000000)
    # Carga del numero de la dimension a partir de las regiones
    n_dimension = regiones['n_dimension'][0]
    # Carga de los valores eigen significativos de las regiones
    valores_eigen = eig.cargar_valores_eigen(regiones)
    # Calculo de los vectores de los valores eigen
    vectores_valores_eigen = eig.calcular_vectores_valores_eigen(valores_eigen, n_dimension)
    matrices_diagonales_valores_eig = eig.calcular_matrices_diagonal_valores_eigen(vectores_valores_eigen, n_dimension)
    # Se cargan las funciones hiperbolicas
    funciones_hiperbolicas = hyp.cargar_funciones_hiperbolicas()
    # Se crean dos vectores uno sin error y el otro con error
    vectores_funciones_hiperbolicas = hyp.calcular_vectores_funciones_hiperbolicas(funciones_hiperbolicas, vectores_valores_eigen, n_dimension)
    matrices_diagonales_hiperbolicas = hyp.calcular_matrices_diagonales_hiperbolicas(vectores_funciones_hiperbolicas, n_dimension)
    # Se carga la informacion necesaria para poder calcular las matrices de acoplamiento
    integrandos_matrices_acoplamiento = m_acop.cargar_integrandos_matrices_acoplamiento()
    # Se calcula las matrices con la solucion analitica
    matrices_acomplamiento_sol = m_acop.calcular_matrices_acomplamiento_solucion(integrandos_matrices_acoplamiento, vectores_valores_eigen, n_dimension)
    matrices_acomplamiento_trans = m_acop.calcular_matrices_acomplamiento_transpuestas(matrices_acomplamiento_sol)
    # Se carga la informacion necesaria para poder calcular los vectores distorsionadores
    integrandos_vectores_distorsionadores = v_dist.cargar_integrandos_vectores_distorsionadores()
    # Se calcula los vectores distorsionadores con quad de scipy se hace una muestra para solo 10 valores eigen de cada vector eigen
    vectores_distorsionadores_sol = v_dist.calcular_vectores_distorsionadores_solucion(integrandos_vectores_distorsionadores, vectores_valores_eigen, n_dimension)
    # ordenar_constantes()
    constantes_c = calcular_matriz_gauss(vectores_valores_eigen, matrices_diagonales_valores_eig, matrices_diagonales_hiperbolicas, matrices_acomplamiento_sol,
                          vectores_distorsionadores_sol, matrices_acomplamiento_trans, n_dimension)
    print("\n\t\tConstantes C calculadas\n")
    print(constantes_c)
    print("\n\t\tConstante C1\n")
    print(constantes_c['C1'].to_numpy().shape)
if __name__ == '__main__':
	main()
