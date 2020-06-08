import numpy as np
import pandas as pd
from numpy import pi

from config import conf
import region

__doc__ = """
En este paquete se definen funciones para crear y calcular todo lo que tenga
que ver con valores eigen, vectores eigen , matrices de valores eigen y refe
rente.
Este modulo es equivalente a: f03_Valor_Eigen(reginoN, m) hasta f04_Diag_Valor_
Eigen_Prueba( ). Nota general: Solo el modulo graficacion.py es el que contiene
las pruebas o graficas respectivas a cada modulo.
"""


def cargar_valores_eigen(regiones):
    """
    Funcion encargada de cargar los valores eigen representativos de todas las
    regiones del problema plano, es decir, si hay valores repetidos,esta funcion
    solo cargara un valor eigen representativo. Recibe como parametro de entrada
    el DataFrame regiones.

    Retorna:
    * valores_eigen: DataFrame encargado de guardar los valores eigen representativos
      de todas las regiones y de como calcularlos.
      Contiene:
          * xi: x inicial
          * xf: y inicial
          * tipo de funcion: seno o coseno
          * calcular: Almacena como se deben calcular el valor eigen dado
          * calcular_err: Almacena como se  debe calcular el valor eigen con un
            error dado.
          * calcular_str: Almacena como se calcula el valor eigen en una variable
            tipo string, para efectos de graficacion

    Ejemplo de salida valores_eigen DataFrame:
            xi  xf  tipo            calcular           calcular_err               calcular_str

        P   2   4   sin          (m*pi)/(2)    (m*pi)/((error*4)-2)             (m*pi)/(x4-x2)
        Q   0   6   cos  ((2*m-1)*pi)/(2*6)    ((2*m-1)*pi)/(2*((error*6)-0))  ((2*m-1)*pi)/(2*(x6-x0))
        R   0   1   cos  ((2*m-1)*pi)/(2*1)    ((2*m-1)*pi)/(2*((error*1)-0))  ((2*m-1)*pi)/(2*(x1-x0))
        W   1   2   sin          (m*pi)/(1)    (m*pi)/((error*2)-1)            (m*pi)/(x2-x1)
        Z   2   3   sin          (m*pi)/(1)    (m*pi)/((error*3)-2)            (m*pi)/(x3-x2)
        T   3   5   sin          (m*pi)/(2)    (m*pi)/((error*5)-3)            (m*pi)/(x5-x3)
        U   0   6   sin          (m*pi)/(6)    (m*pi)/((error*6)-0)            (m*pi)/(x6-x0)
    """
    # Se cargan los index es decir 'P, Q,...,Z' significativos (no repetidos)
    valores_index_significativos = regiones['chr_eigen'].drop_duplicates().index
    # Se crea el DataFrame a partir de los valores eigen significativos
    valores_eigen = pd.DataFrame({'xi': regiones['xi'].reindex(valores_index_significativos),
                                  'xf': regiones['xf'].reindex(valores_index_significativos),
                                  'tipo': regiones['f_eigen'].reindex(valores_index_significativos)}, index=valores_index_significativos)
    # Transformar index en: P, Q,...., Z (Nombre de los Valores eigen)
    valores_eigen.index = regiones['chr_eigen'].drop_duplicates()
    # DEBUG print(regiones['chr_eigen'].drop_duplicates())
    # Eliminar el nombre 'chr_eigen' del index
    valores_eigen.index.name = ""
    # Agregar como se calculan los valores eigen
    valores_eigen.loc[valores_eigen['tipo'] == 'sin', 'calcular'] = '(m*pi)/(' + (valores_eigen['xf'] - valores_eigen['xi']).astype(str) + ')'
    valores_eigen.loc[valores_eigen['tipo'] == 'cos', 'calcular'] = '((2*m-1)*pi)/(2*' + (valores_eigen['xf'] - valores_eigen['xi']).astype(str) + ')'
    # Agregar como se calculan los valores eigen con error
    valores_eigen.loc[valores_eigen['tipo'] == 'sin', 'calcular_err'] = '(m*pi)/((error*' + valores_eigen['xf'].astype(str) + ')-' + valores_eigen['xi'].astype(str) + ')'
    valores_eigen.loc[valores_eigen['tipo'] == 'cos', 'calcular_err'] = '((2*m-1)*pi)/(2*((error*' + valores_eigen['xf'].astype(str) + ')-' + valores_eigen['xi'].astype(str) + '))'
    # Agregar como se calculan los valores eigen en cadena str para efectos de graficacion
    valores_eigen.loc[valores_eigen['tipo'] == 'sin', 'calcular_str'] = '(m*pi)/(x' + valores_eigen['xf'].astype(str) + '-x' + valores_eigen['xi'].astype(str) + ')'
    valores_eigen.loc[valores_eigen['tipo'] == 'cos', 'calcular_str'] = '((2*m-1)*pi)/(2*(x' + valores_eigen['xf'].astype(str) + '-x' + valores_eigen['xi'].astype(str) + '))'
    return valores_eigen


def calcular_vectores_valores_eigen(valores_eigen, n_dimension):
    """
    Funcion encargada de calcular los vectores de valores eigen segun la
    dimension dada.
    Parametros de entrada:
        * valores_eigen: DataFrame encargado de guardar los valores eigen
          representativos de todas las regiones y de como calcularlos.
        * n_dimension: Dimension a utilizar en la creacion de los vectores,
          matrices y demas.

    Salida:
        * vectores_valores_eigen: Es un DataFrame creado a partir de los
          valores eigen dados.

    Nota: Para un ejemplo dado remitirse a la funcion main de eigen.py.
    """
    # Esta matriz m funciona para la operacion matricial de como calcula el vector eigen
    # Es la que hace de contador desde m = 1, 2, 3,..., n_dimension
    m = np.arange(1, n_dimension + 1)
    # Se crea el DataFrame de dimension = valores_eigen*n_dimension y se llena con ceros
    vectores_valores_eigen = pd.DataFrame(np.zeros(((len(valores_eigen.index)), n_dimension)), index=valores_eigen.index)
    # Para evitar confusiones los indices de las columnas van desde 1 en adelante
    vectores_valores_eigen.columns += 1
    # ch_eigen -> 'P','Q','...','Z'
    for chr_eigen in valores_eigen.index:
        # Se evalua la expresion completa a traves de cada valor eigen
        vectores_valores_eigen.T[chr_eigen] = pd.eval(valores_eigen.loc[chr_eigen, 'calcular'])
        # print(valores_eigen.loc[chr_eigen, 'calcular'])
    # Para llevar los valores eigen a salida csv pandas
    #vectores_valores_eigen.T.to_csv('csv/' + conf.data['env']['path'] + '/vectores_valores_eigen.csv')
    return vectores_valores_eigen


def calcular_vectores_valores_eigen_error(valores_eigen, n_dimension, error=1):
    """
    Funcion encargada de calcular los vectores de valores eigen dado un error,
    es decir se agrega un error en el calculo de los vectores.
    Parametros de entrada:
        * valores_eigen: DataFrame encargado de guardar los valores eigen
          representativos de todas las regiones y de como calcularlos.
        * n_dimension: Dimension a utilizar en la creacion de los vectores,
          matrices y demas.
        * error: El error introducido en el calculo de los vectores

    Salida:
        * vectores_valores_eigen: Es un DataFrame creado a partir de los valores
        eigen y el error dados.

    Nota: Para un ejemplo dado remitirse a la funcion main de eigen.py.
    """
    # Esta matriz m funciona para la operacion matricial de como calcula el vector eigen
    # Es la que hace de contador desde m = 1, 2, 3,..., n_dimension
    m = np.arange(1, n_dimension + 1)
    vectores_valores_eigen_error = pd.DataFrame(np.zeros(((len(valores_eigen.index)), n_dimension)), index=valores_eigen.index)
    vectores_valores_eigen_error.columns += 1
    for chr_eigen in valores_eigen.index:
        # Se evalua la expresion completa a traves de cada valor eigen
        vectores_valores_eigen_error.T[chr_eigen] = pd.eval(valores_eigen.loc[chr_eigen, 'calcular_err'])
    # vectores_valores_eigen_comparacion.T.to_csv('csv/vectores_valores_eigen_comparacion.csv')  # Por si se deseara cargar los vectores a un archivo csv
    # DEBUG print(f"El valor maximo del vectores_valores_eigen_comparacion es: {max(vectores_valores_eigen_comparacion)}")
    return vectores_valores_eigen_error


def calcular_matrices_diagonal_valores_eigen(vectores_valores_eigen, n_dimension):
    """
    Se determina las matrices diagonales de los valores eigen
    """
    # Se crea un Multindex para poder almacenar los datos con 3 dimensiones
    # esto es 'Pn' = Matriz diagonal de valores eigen Pn , 'Qn' = Matriz diagonal de valores eigen Qn ,....,
    # 'Zn' = Matriz diagonal de valores eigen Zn
    # El index queda de la forma = [('Pn',1), ('Pn',2), ('Pn',3),..., ('Un',99), ('Un',100)]
    index = pd.MultiIndex.from_tuples([(chr_eigen, i) for chr_eigen in vectores_valores_eigen.index for i in range(1, n_dimension + 1)])
    # Se crea el dataframe de matrices diagonales de valores eigen
    matrices_diagonales_v_eig = vectores_valores_eigen.reindex(index)
    # Se llena matriz a matriz con su respectiva diagonal
    # print(f"\t\tINSIDE Matrices Diagonales")
    # print(matrices_diagonales_v_eig)
    for chr_eigen in vectores_valores_eigen.index:
        matrices_diagonales_v_eig.loc[chr_eigen] = np.diag(vectores_valores_eigen.loc[chr_eigen])
    return matrices_diagonales_v_eig


def calcular_matrices_diagonal_valores_eigen_error(matrices_diagonales_v_eig, n_dimension=100, error=1):
    """
    Se determina las matrices diagonales de los valores eigen con error
    Parametros de entrada:
        * matrices_diagonales_v_eig: matrices diagonales de los valores eigen
        * n_dimension: Dimension a utilizar en la creacion de los vectores,
          matrices y demas.

    Salida:
        * DataFrame matrices diagonales de valores eigen con error
    """
    return (error * matrices_diagonales_v_eig.abs())


def main():
    config.init(problema_plano='ejemplo')    
    # Se crea el df de regiones
    regiones = region.cargar_regiones(10)  # 20000000
    # Se crea variable local n_dimension a partir de n_dimension  obtenida del df regiones
    n_dimension = regiones['n_dimension'][0]
    valores_eigen = cargar_valores_eigen(regiones)
    vectores_valores_eigen = calcular_vectores_valores_eigen(valores_eigen, n_dimension)
    matrices_diagonales_v_eig = calcular_matrices_diagonal_valores_eigen(vectores_valores_eigen, n_dimension)
    print(regiones)
    # La funcion df.to_string() permite vizualizar todo el df en vez de la cabeza y la cola
    print(valores_eigen.to_string())
    print(f"\t\t\tVectores Eigen")
    print(vectores_valores_eigen.loc[:,1:15])
    vectores_valores_eigen_comparacion = calcular_vectores_valores_eigen_error(valores_eigen, n_dimension, 2)
    print(f"\t\t\tVectores Eigen con error")
    print(vectores_valores_eigen_comparacion)
    print(f"\t\t\tMatrices Diagonales Valores Eigen")
    print(matrices_diagonales_v_eig)
    matrices_diagonales_v_eig_comparacion = calcular_matrices_diagonal_valores_eigen_error(matrices_diagonales_v_eig, n_dimension, 2)
    print(f"\t\t\tMatrices Diagonales Valores Eigen con error")
    print(matrices_diagonales_v_eig_comparacion)
    # ----------------------------------- DEBUG TEMPORAL --------------------------
    print(f"\n{'-'*35}DEBUG{'-'*35}")
    vector_eig = vectores_valores_eigen.loc['P'].to_numpy()
    print(len(vector_eig))


if __name__ == '__main__':
    main()
