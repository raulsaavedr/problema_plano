import numpy as np
import pandas as pd
from numpy import tanh
import logging
from numba import jit

from config import conf

logger = logging.getLogger(__name__)
logger.setLevel(conf.get_logging_level('hiperbolica'))
logger.disabled = conf.data['debugging']['hiperbolica']['disabled']

import region as reg
import eigen as eig

__doc__ = """
En este modulo se encuentran ubicadas todas las funciones referentes a los cal-
culos de las funciones hiperbólicas, es decir, calculo de vectores para luego
calcular matrices con dichas funciones hiperbólicas.
Este modulo es equivalente a: f05_Diag_Func_Hiper() hasta f05_Diag_Func_Hiper_
Prueba( ). Nota general: Solo el modulo graficacion.py es el que contiene
las pruebas o graficas respectivas a cada modulo.
"""

# Se define una funcion cotangente hiperbolicas
@jit
def coth(x):
    """
    Definicion de la funcion cotangente hiperbolica, se hace con la libreria
    numexpr y numpy para que el calculo sea ejecutado de forma eficiente y rapida.
    """
    return (1 / tanh(x))


def cargar_funciones_hiperbolicas():
    """
    Carga en un DataFrame las variables necesarias para calcular los vectores y
    matrices de funciones hiperbolicas. El archivo funciones_hiperbolicas.csv
    esta definido para cada una de las matrices diagonales de funciones hiperbolicas
    y contiene lo siguiente por cada matriz:
        * chr_eigen: El nombre del vector eigen en cuestion(P, Q,..., Z)
        * yi: y inicial
        * yf: y final
        * tipo: funcion tanh o coth
        * divisor: puede ser 1 o 2 depende de cada matriz

        La expresion es contruida de la siguiente forma:
            Di = tipo(chr_eigen*(yf-yi)/divisor)

    Salida:
        DataFrame creado a partir del archivo csv , ademas se agrega una columna
        yn, esta almacena la operacion (yf-yi)/divisor , es usada mas tarde en
        el calculo de los vectores. Ademas se agrega otra columna calcular_str
        en esta se almacena como sera representado el calculo de la diagonal en
        forma de cadena, que es usada luego para el texto de la graficacion.

    Ejemplo de salida del df. de funciones_hiperbolicas:
          chr_eigen	yi	yf	tipo   divisor  yn	  calcular_str
        D1  	P	0	1	coth       1	1.0	  coth(P*(y1-y0)/1)
        D2  	Q	1	2	coth	   2	0.5	  coth(Q*(y2-y1)/2)
        D3  	Q	1	2	tanh	   2	0.5	  tanh(Q*(y2-y1)/2)
        D4  	Q	2	3	coth	   2	0.5	  coth(Q*(y3-y2)/2)
        D5  	Q	2	3	tanh	   2	0.5	  tanh(Q*(y3-y2)/2)
        D6  	P	3	4	coth	   2	0.5	  coth(P*(y4-y3)/2)
        D7  	P	3	4	tanh	   2	0.5	  tanh(P*(y4-y3)/2)
        D8  	Q	4	5	coth	   2	0.5	  coth(Q*(y5-y4)/2)
        D9  	Q	4	5	tanh	   2	0.5	  tanh(Q*(y5-y4)/2)
        D10 	W	5	6	coth	   1	1.0	  coth(W*(y6-y5)/1)
        D11 	T	5	6	coth	   2	0.5	  coth(T*(y6-y5)/2)
        D12 	T	5	6	tanh	   2	0.5	  tanh(T*(y6-y5)/2)
        D13 	R	5	6	coth	   1	1.0	  coth(R*(y6-y5)/1)
        D14 	U	6	7	coth	   1	1.0	  coth(U*(y7-y6)/1)
        D15 	Z	5	6	coth	   1	1.0	  coth(Z*(y6-y5)/1)
    """
    filename = 'csv/' + conf.data['env']['path'] + '/funciones_hiperbolicas.csv'
    funciones_hiperbolicas = pd.read_csv(filename)
    # Se cambia el index a D1, D2,..., DN
    funciones_hiperbolicas.index = ['D'+str(i) for i in range(1, len(funciones_hiperbolicas.index) + 1)]
    # Se calcula yn con base a los yi, yf y divisor dados
    funciones_hiperbolicas['yn'] = (funciones_hiperbolicas['yf'] - funciones_hiperbolicas['yi'])/ funciones_hiperbolicas['divisor']
    # Se construye la expresion string de como es calculado la funcion hiperbolica  # Esto sirve para quitar la n de Pn,Qn,... y queda P,Q,T,...,Z
    funciones_hiperbolicas['calcular_str'] = funciones_hiperbolicas['tipo'] + '(' + funciones_hiperbolicas['chr_eigen'].str.split(pat='n').str[0]  + '*'\
                                             + '(y' + funciones_hiperbolicas['yf'].astype(str) + '-y' + funciones_hiperbolicas['yi'].astype(str) \
                                             + ')/' + funciones_hiperbolicas['divisor'].astype(str) + ')'
    return funciones_hiperbolicas


def calcular_vectores_funciones_hiperbolicas(funciones_hiperbolicas, vectores_valores_eigen, n_dimension=100, error=1):
    """
    Crea un DataFrame a partir de funciones_hiperbolicas, dicho df se llena de
    la siguiente forma:
        * Se calcula el argumento de cada vector diagonal, es decir,
           (chr_eigen*(yf-yi)/divisor) , para el ejemplo dado sería de la forma:
           (P*yn*error), (Q*yn*error),...,(Z*yn*error). Por defecto error=1
        * Se hace busqueda dentro del df funciones_hiperbolicas para saber de
           que tipo es el vector (coth o tanh) y luego aplicarle la funcion
           respectiva al argumento calculado.

    Parametros de entrada:
        * funciones_hiperbolicas: DataFrame donde se almacena como se debe cal-
          cular los vectores o matrices de func. hiperbolicas.
        * vectores_valores_eigen: Es un DataFrame que contiene todos los valores
          eigen de todos los vectores.

    Salida:
        * vectores_funciones_hiperbolicas: Es un DataFrame que contiene todos
          los valores de las funciones hiperbolicas de todos los vectores.
    """
    # Esta matriz m funciona para la operacion matricial de como calcula el vector eigen
    # Es la que hace de contador desde m = 1, 2, 3,..., n_dimension
    m = np.arange(1, n_dimension + 1)
    # Se crea el DataFrame de dimension = valores_eigen*n_dimension
    vectores_funciones_hiperbolicas = pd.DataFrame(np.zeros(((len(funciones_hiperbolicas.index)), n_dimension)), index=funciones_hiperbolicas.index)
    # Para evitar confusiones los indices de las columnas van desde 1 en adelante
    vectores_funciones_hiperbolicas.columns += 1
    # nro_diagonal 'D1', 'D2', ..., 'DN'
    for nro_diagonal in funciones_hiperbolicas.index:
        # Se calcula primero el argumento de los vectores de funciones hiperbolicas
        vectores_funciones_hiperbolicas.loc[nro_diagonal] = vectores_valores_eigen.loc[funciones_hiperbolicas.loc[nro_diagonal,'chr_eigen']]\
                                                            * funciones_hiperbolicas.loc[nro_diagonal, 'yn'] * error
    # Se procede a aplicar la funcion adecuada (tanh o coth )a cada uno de los valores de cada vector
    vectores_funciones_hiperbolicas.loc[funciones_hiperbolicas['tipo']=='coth',:] = coth(vectores_funciones_hiperbolicas.loc[funciones_hiperbolicas['tipo']=='coth',:].to_numpy())
    vectores_funciones_hiperbolicas.loc[funciones_hiperbolicas['tipo']=='tanh',:] = tanh(vectores_funciones_hiperbolicas.loc[funciones_hiperbolicas['tipo']=='tanh',:].to_numpy())
    return vectores_funciones_hiperbolicas


def calcular_matrices_diagonales_hiperbolicas(vectores_funciones_hiperbolicas, n_dimension=100):
    """
    Se determina las matrices diagonales de las funciones hiperbolicas.
    Esta funcion es equivalente a: f05_Diag_Func_Hiper()
    Parametros de entrada:
        * vectores_funciones_hiperbolicas: Es un DataFrame que contiene todos
          los valores de las funciones hiperbolicas de todos los vectores.
        * n_dimension: Dimension a utilizar en la creacion de los vectores,
          matrices y demas.

    Salida:
        * matrices_diagonales_hiperbolicas: un DataFrame de la forma:
                      1         2         3         4    5    ...  96   97   98   99   100
        D1  1    1.090331  0.000000  0.000000  0.000000  0.0  ...  0.0  0.0  0.0  0.0  0.0
            2    0.000000  1.003742  0.000000  0.000000  0.0  ...  0.0  0.0  0.0  0.0  0.0
            3    0.000000  0.000000  1.000161  0.000000  0.0  ...  0.0  0.0  0.0  0.0  0.0
            4    0.000000  0.000000  0.000000  1.000007  0.0  ...  0.0  0.0  0.0  0.0  0.0
            5    0.000000  0.000000  0.000000  0.000000  1.0  ...  0.0  0.0  0.0  0.0  0.0
        ...           ...       ...       ...       ...  ...  ...  ...  ...  ...  ...  ...
    """
    # Se crea un Multindex para poder almacenar los datos con 3 dimensiones
    # esto es 'D1' = Matriz diagonal de funciones hiperbolicas , 'D2' = Matriz diagonal funciones hiperbolicas ,....,
    # 'DN' = funciones hiperbolicas
    # El index queda de la forma = [('D1',1), ('D1',2), ('D1',3),..., ('D15',99), ('D15',100)]
    index = pd.MultiIndex.from_tuples([(nro_diagonal, i) for nro_diagonal in vectores_funciones_hiperbolicas.index for i in range(1, n_dimension + 1)])
    # Se crea el dataframe de matrices diagonales de valores eigen
    matrices_diagonales_hiperbolicas = vectores_funciones_hiperbolicas.reindex(index)
    for nro_diagonal in vectores_funciones_hiperbolicas.index:
        # Se calcula la diagonal de cada uno de los vectores involucrados
        matrices_diagonales_hiperbolicas.loc[nro_diagonal] = np.diag(vectores_funciones_hiperbolicas.loc[nro_diagonal])
    return matrices_diagonales_hiperbolicas
