import region as reg
import eigen as eig
import numpy as np
import pandas as pd
from scipy.integrate import quad

# Descripcion del modulo
__doc__ = """
En este modulo se encuentran todas las funciones referentes a los calculos de
las matrices de acoplamiento. Se definen algunas funciones que son necesarias
para poder integrar utilizando quad de scipy.
Este modulo es equivalente a: f05_intM(), f07_M_Control() y parte de
f08_Matrices_Vectores().
Nota general: Solo el modulo graficacion.py es el que contiene las pruebas o
graficas respectivas a cada modulo.
"""

def integrando_sin_sin(x, vector_valor_eigen_1, desp_1, vector_valor_eigen_2, desp_2):
    """
    Funcion integrando sin(vector_valor_eigen_1*(x-desp_1))*sin(vector_valor_eigen_2*(x-desp_2)). El parametro x
    es un parametro implicito y no se declara , este es utilizado internamente
    por la funcion quad de scipy.integrate.
    Parametros de entrada:
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_1[m].
        * desp_1: desplazamiento en x de la primera funcion.
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_2[n].
        * desp_2: desplazamiento en x de la segunda funcion.
    """
    return np.sin(vector_valor_eigen_1 * (x - desp_1)) * np.sin(vector_valor_eigen_2 * (x - desp_2))


def integrando_sin_cos(x, vector_valor_eigen_1, desp_1, vector_valor_eigen_2, desp_2):
    """
    Funcion integrando sin(vector_valor_eigen_1*(x-desp_1))*cos(vector_valor_eigen_2*(x-desp_2)). El parametro x
    es un parametro implicito y no se declara , este es utilizado internamente
    por la funcion quad de scipy.integrate.
    Parametros de entrada:
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_1[m].
        * desp_1: desplazamiento en x de la primera funcion.
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_2[n].
        * desp_2: desplazamiento en x de la segunda funcion.
    """
    return np.sin(vector_valor_eigen_1 * (x - desp_1)) * np.cos(vector_valor_eigen_2 * (x - desp_2))


def integrando_cos_sin(x, vector_valor_eigen_1, desp_1, vector_valor_eigen_2, desp_2):
    """
    Funcion integrando cos(vector_valor_eigen_1*(x-desp_1))*sin(vector_valor_eigen_2*(x-desp_2)). El parametro x
    es un parametro implicito y no se declara , este es utilizado internamente
    por la funcion quad de scipy.integrate.
    Parametros de entrada:
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_1[m].
        * desp_1: desplazamiento en x de la primera funcion.
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_2[n].
        * desp_2: desplazamiento en x de la segunda funcion.
    """
    return np.cos(vector_valor_eigen_1 * (x - desp_1)) * np.sin(vector_valor_eigen_2 * (x - desp_2))


def integrando_cos_cos(x, vector_valor_eigen_1, desp_1, vector_valor_eigen_2, desp_2):
    """
    Funcion integrando cos(vector_valor_eigen_1*(x-desp_1))*cos(vector_valor_eigen_2*(x-desp_2)). El parametro x
    es un parametro implicito y no se declara , este es utilizado internamente
    por la funcion quad de scipy.integrate.
    Parametros de entrada:
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_1[m].
        * desp_1: desplazamiento en x de la primera funcion.
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_2[n].
        * desp_2: desplazamiento en x de la segunda funcion.
    """
    return np.cos(vector_valor_eigen_1 * (x - desp_1)) * np.cos(vector_valor_eigen_2 * (x - desp_2))


def integrar_func(func_str, xi, xf, vector_valor_eigen_1, desp_1, vector_valor_eigen_2, desp_2):
    """
    Paramentros de entrada:
        * func_str : es la funcion a ser llamada y se almacena en el DataFrame
          integrandos_matrices_acoplamiento['tipo_f'], esta cadena luego sera
          evaluada por la funcion de python eval() para ser pasada como parametro
           a la funcion quad y poder llamar cualquiera de las cuatros funciones
          integrando_ ya declaradas(sin_sin, sin_cos, cos_sin, cos_cos).
        * xi : limite de integracion inicial.
        * xf: limite de integracion final.
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_1[m].
        * desp_1: desplazamiento en x de la primera funcion.
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_2[n].
        * desp_2: desplazamiento en x de la segunda funcion.
    """
    return quad(eval(func_str), xi, xf, args=(vector_valor_eigen_1, desp_1, vector_valor_eigen_2, desp_2), limit=100000)[0]


def sinus(k_veigens, k_veigens_desp, xi, xf):
    """
    Funcion utilizada por las funciones sin_cos y cos_sin , ayuda a calcular
    de forma analitica las integrales sin*cos y cos*sin de xi a xf.
    Parametros de entrada:
        * k_veigens: Vector de constantes calculado a partir de los valores ei-
          gen, esto puede ser:
              k_vectores_valores_eigen_1 = vector_valor_eigen_1 + vector_valor_eigen_2
              k_vectores_valores_eigen_2 = vector_valor_eigen_1 - vector_valor_eigen_2
        * k_veigens_desp: Vector de constantes calculado a partir de los valores
          eigen y los desplazamientos en x dados, esto puede ser:
              k_vectores_valores_eigen_desp_a = (vector_valor_eigen_1 * desp_1) + (vector_valor_eigen_1 * desp_2)
              k_vectores_valores_eigen_desp_b = (vector_valor_eigen_1 * desp_1) - (vector_valor_eigen_1 * desp_2)
    """
    # Este array define un cero relativo para que la funcion pueda ser calculada
    cero_relativo = np.full_like(k_veigens, 1e-100, dtype=float)
    # Con np.where se establece lo siguiente: donde hay un valor diferente de cero
    # entonces permanezca dicho valor, si es igual a cero entonces sustituya por cero_relativo
    k_veigens = np.where(k_veigens != 0, k_veigens, cero_relativo)
    # Se crean las variables temporales donde se almacena la operacion evaluada
    # con los desplazamientos en x
    cos_k_veigens_desp = - (np.cos(k_veigens_desp) / k_veigens) * (np.cos(k_veigens * xf) - np.cos(k_veigens * xi))
    sen_k_veigens_desp = (np.sin(k_veigens_desp) / k_veigens) * (np.sin(k_veigens * xf) - np.sin(k_veigens * xi))
    return 0.5 * (cos_k_veigens_desp - sen_k_veigens_desp)


def cosinus(k_veigens, k_veigens_desp, xi, xf):
    """
    Funcion utilizada por las funciones sin_sin y cos_cos , ayuda a calcular
    de forma analitica las integrales sin*sin y cos*cos de xi a xf.
    Parametros de entrada:
        * k_veigens: Vector de constantes calculado a partir de los valores ei-
          gen, esto puede ser:
              k_vectores_valores_eigen_1 = vector_valor_eigen_1 + vector_valor_eigen_2
              k_vectores_valores_eigen_2 = vector_valor_eigen_1 - vector_valor_eigen_2
        * k_veigens_desp: Vector de constantes calculado a partir de los valores
          eigen y los desplazamientos en x dados, esto puede ser:
              k_vectores_valores_eigen_desp_a = (vector_valor_eigen_1 * desp_1) + (vector_valor_eigen_1 * desp_2)
              k_vectores_valores_eigen_desp_b = (vector_valor_eigen_1 * desp_1) - (vector_valor_eigen_1 * desp_2)
    """
    # Este array define un cero relativo para que la funcion pueda ser calculada
    cero_relativo = np.full_like(k_veigens, 1e-100, dtype=float)
    # Con np.where se establece lo siguiente: donde hay un valor diferente de cero
    # entonces permanezca dicho valor, si es igual a cero entonces sustituya por cero_relativo
    k_veigens = np.where(k_veigens != 0, k_veigens, cero_relativo)
    # Se crean las variables temporales donde se almacena la operacion evaluada
    # con los desplazamientos en x
    cos_k_veigens_desp = (np.cos(k_veigens_desp) / k_veigens) * (np.sin(k_veigens * xf) - np.sin(k_veigens * xi))
    sen_k_veigens_desp = (np.sin(k_veigens_desp) / k_veigens) * (np.cos(k_veigens * xf) - np.cos(k_veigens * xi))
    return 0.5 * (cos_k_veigens_desp - sen_k_veigens_desp)


def sin_sin(xi, xf, vector_valor_eigen_1, desp_1, vector_valor_eigen_2, desp_2):
    """
    Funcion que evalua el resultado de forma analitica de la integral:
        sin(vector_valor_eigen_1*(x-desp_1))*sin(vector_valor_eigen_2*(x-desp_2)) de xi a xf
    Parametros de Entrada:
        * xi : limite de integracion inicial
        * xf: limite de integracion final
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_1[m]
        * desp_1: desplazamiento en x de la primera funcion
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_2[n]
        * desp_2: desplazamiento en x de la segunda funcion
    """
    k_vectores_valores_eigen_1 = vector_valor_eigen_1 + vector_valor_eigen_2
    k_vectores_valores_eigen_2 = vector_valor_eigen_1 - vector_valor_eigen_2
    k_vectores_valores_eigen_desp_a = (vector_valor_eigen_1 * desp_1) + (vector_valor_eigen_1 * desp_2)
    k_vectores_valores_eigen_desp_b = (vector_valor_eigen_1 * desp_1) - (vector_valor_eigen_1 * desp_2)
    return (cosinus(k_vectores_valores_eigen_2, k_vectores_valores_eigen_desp_b, xi, xf) - cosinus(k_vectores_valores_eigen_1, k_vectores_valores_eigen_desp_a, xi, xf))

def sin_cos(xi, xf, vector_valor_eigen_1, desp_1, vector_valor_eigen_2, desp_2):
    """
    Funcion que evalua el resultado de forma analitca de la integral:
        sin(vector_valor_eigen_1*(x-desp_1))*cos(vector_valor_eigen_2*(x-desp_2)) de xi a xf
    Parametros de Entrada:
        * xi : limite de integracion inicial
        * xf: limite de integracion final
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_1[m]
        * desp_1: desplazamiento en x de la primera funcion
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_2[n]
        * desp_2: desplazamiento en x de la segunda funcion
    """
    k_vectores_valores_eigen_1 = vector_valor_eigen_1 + vector_valor_eigen_2
    k_vectores_valores_eigen_2 = vector_valor_eigen_1 - vector_valor_eigen_2
    k_vectores_valores_eigen_desp_a = (vector_valor_eigen_1 * desp_1) + (vector_valor_eigen_1 * desp_2)
    k_vectores_valores_eigen_desp_b = (vector_valor_eigen_1 * desp_1) - (vector_valor_eigen_1 * desp_2)
    return (sinus(k_vectores_valores_eigen_2, k_vectores_valores_eigen_desp_b, xi, xf) + sinus(k_vectores_valores_eigen_1, k_vectores_valores_eigen_desp_a, xi, xf))


def cos_sin(xi, xf, vector_valor_eigen_1, desp_1, vector_valor_eigen_2, desp_2):
    """
    Funcion que evalua el resultado de forma analitca de la integral:
        cos(vector_valor_eigen_1*(x-desp_1))*sin(vector_valor_eigen_2*(x-desp_2)) de xi a xf
    Parametros de Entrada:
        * xi : limite de integracion inicial
        * xf: limite de integracion final
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_1[m]
        * desp_1: desplazamiento en x de la primera funcion
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_2[n]
        * desp_2: desplazamiento en x de la segunda funcion
    """
    k_vectores_valores_eigen_1 = vector_valor_eigen_1 + vector_valor_eigen_2
    k_vectores_valores_eigen_2 = vector_valor_eigen_1 - vector_valor_eigen_2
    k_vectores_valores_eigen_desp_a = (vector_valor_eigen_1 * desp_1) + (vector_valor_eigen_1 * desp_2)
    k_vectores_valores_eigen_desp_b = (vector_valor_eigen_1 * desp_1) - (vector_valor_eigen_1 * desp_2)
    return (- sinus(k_vectores_valores_eigen_2, k_vectores_valores_eigen_desp_b, xi, xf) + sinus(k_vectores_valores_eigen_1, k_vectores_valores_eigen_desp_a, xi, xf))


def cos_cos(xi, xf, vector_valor_eigen_1, desp_1, vector_valor_eigen_2, desp_2):
    """
    Funcion que evalua el resultado de forma analitca de la integral:
        cos(vector_valor_eigen_1*(x-desp_1))*cos(vector_valor_eigen_2*(x-desp_2)) de xi a xf
    Parametros de Entrada:
        * xi : limite de integracion inicial
        * xf: limite de integracion final
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_1[m]
        * desp_1: desplazamiento en x de la primera funcion
        * vector_valor_eigen_1: es el valor eigen en la posicion vector_valor_eigen_2[n]
        * desp_2: desplazamiento en x de la segunda funcion
    """
    k_vectores_valores_eigen_1 = vector_valor_eigen_1 + vector_valor_eigen_2
    k_vectores_valores_eigen_2 = vector_valor_eigen_1 - vector_valor_eigen_2
    k_vectores_valores_eigen_desp_a = (vector_valor_eigen_1 * desp_1) + (vector_valor_eigen_1 * desp_2)
    k_vectores_valores_eigen_desp_b = (vector_valor_eigen_1 * desp_1) - (vector_valor_eigen_1 * desp_2)
    return (cosinus(k_vectores_valores_eigen_2, k_vectores_valores_eigen_desp_b, xi, xf) + cosinus(k_vectores_valores_eigen_1, k_vectores_valores_eigen_desp_a, xi, xf))


def cargar_integrandos_matrices_acoplamiento():
    """
    Funcion encargada de cargar en un dataframe lo necesario para contruir como
    se calcularan las matrices cuadradas de acomplamiento. El archivo
    matrices_cuadradas_acomplamiento.csv debe ser definido de la siguiente
    manera:
        La primera fila se define como la fila de los nombres que tomara cada
    columna, esta fila luce asi:
        xi,xf,tipo_1,chr_eigen_1,desp_1,tipo_2,chr_eigen_2,desp_2

        Donde:
            * xi : limite de integracion inicial.
            * xf: limite de integracion final.
            * tipo_1: tipo de funcion del primer factor (sin o cos)
            * ch_eigen_1: es el vector eigen con que se calculara el primer
              factor.
            * desp_1: desplazamiento en x del primer factor.
            * tipo_2: tipo de funcion del segundo factor (sin o cos)
            * ch_eigen_2: es el vector eigen con que se calculara el segundo
              factor.
            * desp_2: desplazamiento en x del segundo factor.

        Estos valores son dados por fila donde la primera fila despues de la
    fila de nombres de variables representa la matriz numero 1, la siguiente
    la matriz numero dos y asi sucesivamente.

    Salida:
        * DataFrame que contiene todos los valores eigen , tipo de funciones de
          cada factor del integrado y los desplazamiento de cada matriz de aco-
          plamiento.

    Ejemplo de salida del DataFrame integrandos_matrices_acoplamiento:
            xi	xf	tipo_1	chr_eigen_1	desp_1	tipo_2	chr_eigen_2	desp_2	tipo_f	               calcular_str
        M1	2	4	sin	         P   	  2	      sin	     Q	      0	    sin_sin	   Int(sin(Pm*(x-x2))*sin(Qn*(x-x0))) de x2 a x4
        M2	2	6	sin	         Q	      0	      cos	     Q	      0	    sin_cos	   Int(sin(Qm*(x-x0))*cos(Qn*(x-x0))) de x2 a x6
        M3	2	4	sin	         P	      2	      cos	     Q	      0	    sin_cos	   Int(sin(Pm*(x-x2))*cos(Qn*(x-x0))) de x2 a x4
        M4	1	2	sin	         W	      1	      sin	     Q	      0	    sin_sin	   Int(sin(Wm*(x-x1))*sin(Qn*(x-x0))) de x1 a x2
        M5	3	5	sin	         T	      3	      sin	     Q	      0	    sin_sin	   Int(sin(Tm*(x-x3))*sin(Qn*(x-x0))) de x3 a x5
        M6	0	1	cos	         R	      0	      cos	     U	      0	    cos_cos	   Int(cos(Rm*(x-x0))*cos(Un*(x-x0))) de x0 a x1
        M7	2	3	sin	         Z	      2	      cos	     U	      0	    sin_cos	   Int(sin(Zm*(x-x2))*cos(Un*(x-x0))) de x2 a x3
        M8	3	5	sin	         T	      3	      cos	     U	      0	    sin_cos	   Int(sin(Tm*(x-x3))*cos(Un*(x-x0))) de x3 a x5
    """
    integrandos_matrices_acoplamiento = pd.read_csv('csv/matrices_cuadradas_acomplamiento.csv')
    # Se cambian los index de 0,1,2,...ultima matriz a M1,M2,M3,...Mn
    integrandos_matrices_acoplamiento.index = ['M'+str(i) for i in range(1, len(integrandos_matrices_acoplamiento.index) + 1)]
    # Se agregan el tipo de funcion (sin_sin, sin_cos, cos_sin, cos_cos)
    integrandos_matrices_acoplamiento['tipo_f'] = integrandos_matrices_acoplamiento['tipo_1'] + '_'\
                                                  + integrandos_matrices_acoplamiento['tipo_2']
    # Se crea la columna calcular tipo cadena que se utilizara al momento de plotear
    integrandos_matrices_acoplamiento['calcular_str'] = 'Int(' + integrandos_matrices_acoplamiento['tipo_1'] + '(' \
                                                        + integrandos_matrices_acoplamiento['chr_eigen_1']\
                                                        + 'm*(x-x' + integrandos_matrices_acoplamiento['desp_1'].astype(str) + '))*'\
                                                        + integrandos_matrices_acoplamiento['tipo_2'] + '(' \
                                                        + integrandos_matrices_acoplamiento['chr_eigen_2'] + 'n*(x-x'\
                                                        + integrandos_matrices_acoplamiento['desp_2'].astype(str) + ')))'\
                                                        + ' de x' + integrandos_matrices_acoplamiento['xi'].astype(str) + ' a x'\
                                                        + integrandos_matrices_acoplamiento['xf'].astype(str)
    return integrandos_matrices_acoplamiento


def calcular_matrices_acomplamiento_integral(integrandos_matrices_acoplamiento, vectores_valores_eigen, n_dimension=100):
    """
    Funcion encargada de calcular las matrices de acomplamiento a traves de la
    libreria scipy.integrate con la funcion quad.
    Parametros de entrada:
        * integrandos_matrices_acoplamiento: DataFrame que contiene todos los
          valores eigen , tipo de funciones de cada factor del integrando y los
          desplazamiento de cada matriz de acoplamiento.
        * vectores_valores_eigen: DataFrame que contiene todos los vectores
          eigen calculados a partir de los valores eigen.
        * n_dimension: Dimension a utilizar en la creacion de los vectores,
          matrices y demas.

    Salida:
        * matrices_acomplamiento: DataFrame que contiene todas las matrices cua-
          dradas de acoplamiento calculadas con la forma integral.
    """
    # Se vectoriza la funcion de integrar para que sea mucho mas rapida su ejecucion
    integrar_func_vectorized = np.vectorize(integrar_func)
    # Se crea un Multindex para almacenar en un df las matrices generadas
    index = pd.MultiIndex.from_tuples([(M, i) for M in integrandos_matrices_acoplamiento.index for i in range(1, n_dimension + 1)])
    # Se crea el dataframe de matrices de acomplamiento de n_dimension*n_dimension
    matrices_acomplamiento = vectores_valores_eigen.reindex(index)
    print("\n\tCalculando matrices cuadradas de acomplamiento:\n\t\tSolucion con quad de scipy\n")
    for M in integrandos_matrices_acoplamiento.index:
        chr_eigen_1 = integrandos_matrices_acoplamiento.loc[M, 'chr_eigen_1']
        chr_eigen_2 = integrandos_matrices_acoplamiento.loc[M, 'chr_eigen_2']
        tipo_f = integrandos_matrices_acoplamiento.loc[M,'tipo_f']
        xi = integrandos_matrices_acoplamiento.loc[M, 'xi']
        xf = integrandos_matrices_acoplamiento.loc[M, 'xf']
        desp_1 = integrandos_matrices_acoplamiento.loc[M, 'desp_1']
        desp_2 = integrandos_matrices_acoplamiento.loc[M, 'desp_2']
        vector_valor_eigen_1 = vectores_valores_eigen.loc[chr_eigen_1].to_numpy()
        vector_valor_eigen_2 = vectores_valores_eigen.loc[chr_eigen_2].to_numpy()[:,np.newaxis]
        print(f"* Matriz {M} = {integrandos_matrices_acoplamiento.loc[M, 'calcular_str']}")
        matrices_acomplamiento.loc[M] = integrar_func_vectorized('integrando_' + tipo_f,xi,xf,vector_valor_eigen_1, desp_1,vector_valor_eigen_2, desp_2).T
    return matrices_acomplamiento

def calcular_matrices_acomplamiento_solucion(integrandos_matrices_acoplamiento, vectores_valores_eigen, n_dimension=100):
    """
    Funcion encargada de calcular las matrices de acomplamiento a traves de la
    solucion analitica de todos los casos de integrales de matrices cuadradas
    de acoplamiento.
    Parametros de entrada:
        * integrandos_matrices_acoplamiento: DataFrame que contiene todos los
          valores eigen , tipo de funciones de cada factor del integrando y los
          desplazamiento de cada matriz de acoplamiento.
        * vectores_valores_eigen: DataFrame que contiene todos los vectores
          eigen calculados a partir de los valores eigen.
        * n_dimension: Dimension a utilizar en la creacion de los vectores,
          matrices y demas.

    Salida:
        * matrices_acomplamiento: DataFrame que contiene todas las matrices cua-
          dradas de acoplamiento calculadas con la forma analitica.
    """
    index = pd.MultiIndex.from_tuples([(M, i) for M in integrandos_matrices_acoplamiento.index for i in range(1, n_dimension + 1)])
    # Se crea el dataframe de matrices de acomplamiento
    matrices_acomplamiento = vectores_valores_eigen.reindex(index)
    print("\n\tCalculando matrices cuadradas de acomplamiento:\n\t\t\tSolucion analitica\n")
    for M in integrandos_matrices_acoplamiento.index:
        chr_eigen_1 = integrandos_matrices_acoplamiento.loc[M, 'chr_eigen_1']
        chr_eigen_2 = integrandos_matrices_acoplamiento.loc[M, 'chr_eigen_2']
        tipo_f = integrandos_matrices_acoplamiento.loc[M, 'tipo_f']
        xi = integrandos_matrices_acoplamiento.loc[M, 'xi']
        xf = integrandos_matrices_acoplamiento.loc[M, 'xf']
        desp_1 = integrandos_matrices_acoplamiento.loc[M, 'desp_1']
        desp_2 = integrandos_matrices_acoplamiento.loc[M, 'desp_2']
        vector_valor_eigen_1 = vectores_valores_eigen.loc[chr_eigen_1].to_numpy()
        vector_valor_eigen_2 = vectores_valores_eigen.loc[chr_eigen_2].to_numpy()[:, np.newaxis]
        print(f"* Matriz {M} = {integrandos_matrices_acoplamiento.loc[M, 'calcular_str']}")
        matrices_acomplamiento.loc[M] = eval(tipo_f + '(xi, xf, vector_valor_eigen_1, desp_1, vector_valor_eigen_2, desp_2)').T
    return matrices_acomplamiento



def calcular_matrices_acomplamiento_transpuestas(matrices_acomplamiento):
    """
    Funcion que calcula las matrices de acoplamiento transpuestas
    Parametros de entrada:
        * matrices_acomplamiento: DataFrame que contiene todas las
          matrices de acoplamiento
    Salida:
        * matrices_acomplamiento_transpuestas
    """
    # Se genera una copia de las matrices de acoplamiento para reservar el espacio en memoria y preservar el multiindex
    matrices_acomplamiento_transpuestas = matrices_acomplamiento.copy()
    # Se crea el index M1,M2,M3,...Mn
    index = matrices_acomplamiento.index.droplevel(level=1).drop_duplicates()
    for M in index:
        # Se llenan las matrices transpuestas
        matrices_acomplamiento_transpuestas.loc[M] = matrices_acomplamiento.loc[M].T.to_numpy()
    return matrices_acomplamiento_transpuestas


def main():
    regiones = reg.cargar_regiones(10)  # region.cargarRegiones(2000000)
    # Carga de dimension a partir de las regiones
    n_dimension = regiones['n_dimension'][0]
    # Carga de los valores eigen significativos de las regiones
    valores_eigen = eig.cargar_valores_eigen(regiones)
    # Calculo de los vectores de los valores eigen
    vectores_valores_eigen = eig.calcular_vectores_valores_eigen(valores_eigen, n_dimension)
    # Se carga la informacion necesaria para poder calcular las matrices de acoplamiento
    integrandos_matrices_acoplamiento = cargar_integrandos_matrices_acoplamiento()
    # Se calcula las matrices con la funcion quad de scipy
    matrices_acomplamiento_int = calcular_matrices_acomplamiento_integral(integrandos_matrices_acoplamiento, vectores_valores_eigen, n_dimension)
    # Se calcula las matrices con la solucion analitica
    matrices_acomplamiento_sol = calcular_matrices_acomplamiento_solucion(integrandos_matrices_acoplamiento, vectores_valores_eigen, n_dimension)
    # Se calcula las matrices acopladoras transpuestas
    matrices_acomplamiento_trans = calcular_matrices_acomplamiento_transpuestas(matrices_acomplamiento_int)
    print(f"\nSolucion calculada con quad\n{matrices_acomplamiento_int}")
    print(f"\nSolucion calculada con analitica\n{matrices_acomplamiento_sol}")
    print(f"\nDiferencia entre soluciones:\n{matrices_acomplamiento_int - matrices_acomplamiento_sol}")
    print(f"\nMatrices transpuestas de acoplamiento\n{matrices_acomplamiento_trans}")


if __name__=='__main__':
    main()
