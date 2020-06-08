import numpy as np
import pandas as pd
from scipy.integrate import quad
from numba import jit

from config import conf
import region as reg
import eigen as eig
# Descripcion del modulo
__doc__ = """
En este modulo se encuentran todas las funciones referentes a los calculos de
los vectores distorsionadores, es decir, calculo de integrales con soluciones
analiticas y soluciones a traves de metodos numericos.
Este modulo es equivalente a: f07_intS(), f07_S_Control() y parte de
f08_Matrices_Vectores().
Nota general: Solo el modulo graficacion.py es el que contiene las pruebas o
graficas respectivas a cada modulo.
"""
@jit
def integrando_sin(x, vector_valor_eigen, desp_ca, desp_cb):
    """
    Funcion integrando sin(vector_valor_eigen * (x - desp_ca)). El parametro x
    es un parametro implicito y no se declara , este es utilizado internamente
    por la funcion quad de scipy.integrate.
    Parametros de entrada:
        * vector_valor_eigen: es el valor eigen en la posicion valoreigen[m].
        * desp_ca: desplazamiento en x de la funcion.
    """
    return np.sin(vector_valor_eigen * (x - desp_ca))

@jit
def integrando_cos(x, vector_valor_eigen, desp_ca, desp_cb):
    """
    Funcion integrando cos(vector_valor_eigen * (x - desp_ca)). El parametro x
    es un parametro implicito y no se declara , este es utilizado internamente
    por la funcion quad de scipy.integrate.
    Parametros de entrada:
        * vector_valor_eigen: es el valor eigen en la posicion valoreigen[m].
        * desp_ca: desplazamiento en x de la funcion.
    """
    return np.cos(vector_valor_eigen * (x - desp_ca))

@jit
def integrando_por_partes_sin(x, vector_valor_eigen, desp_ca, desp_cb):
    """
    Funcion integrando ((x-desp_ca)/desp_cb-desp_ca)*sin(vector_valor_eigen*(x-desp_ca)).
    El parametro x es un parametro implicito y no se declara , este es
    utilizado internamente por la funcion quad de scipy.integrate.
    Parametros de entrada:
        * vector_valor_eigen: es el valor eigen en la posicion valoreigen[m].
        * desp_ca: desplazamiento en x constante a
        * desp_cb: desplazamiento en x constante b
    """
    return ((x - desp_ca) / (desp_cb - desp_ca)) * np.sin(vector_valor_eigen * x)

@jit
def integrando_por_partes_cos(x, vector_valor_eigen, desp_ca, desp_cb):
    """
    Funcion integrando ((x-desp_ca)/desp_cb-desp_ca)*cos(vector_valor_eigen*(x - desp_ca)).
    El parametro x es un parametro implicito y no se declara , este es
    utilizado internamente por la funcion quad de scipy.integrate.
    Parametros de entrada:
        * vector_valor_eigen: es el valor eigen en la posicion valoreigen[m].
        * desp_ca: desplazamiento en x constante a
        * desp_cb: desplazamiento en x constante b
    """
    return ((x - desp_ca) / (desp_cb - desp_ca)) * np.cos(vector_valor_eigen * x)


def integrar_func(func_str, xi, xf, vector_valor_eigen, desp_ca, desp_cb):
    """
    Funcion encargada de integrar las funciones involucradas con el calculo
    de los vectores distorsionadores.
    Paramentros de entrada:
        * func_str : es la funcion a ser llamada y se almacena en el DataFrame
          integrandos_vectores_distorsionadores['tipo_f'], esta cadena luego sera
          evaluada por la funcion de python eval() para ser pasada como parametro
          a la funcion quad y poder llamar cualquiera de las cuatros funciones
          integrando_ ya declaradas(sin, cos, por_partes_cos, por_partes_cos).
        * xi : limite de integracion inicial.
        * xf: limite de integracion final.
        * vector_valor_eigen: es el valor eigen en la posicion valoreigen[m].
        * desp_ca: es el desplazamiento en x cuando es sin o cos, y cuando es
          integracion por partes representa el x inicial de la region
        * desp_cb: es el x final de la region en tipo de integracion por partes.
    """
    return quad(eval(func_str), xi, xf, args=(vector_valor_eigen,  desp_ca, desp_cb), limit=100000)[0]


def sol_analitica_sin(xi, xf, vector_valor_eigen, desp_ca, desp_cb=0):
    """
    Funcion que evalua el resultado de forma analitica de la integral:
        sin(vector_valor_eigen*(x-desp_ca)) de xi a xf
    Parametros de Entrada:
        * xi : limite de integracion inicial
        * xf: limite de integracion final
        * vector_valor_eigen: es el valor eigen en la posicion vector_valor_eigen[m]
        * desp_ca: es el desplazamiento en x cuando
        * desp_cb: parametro por defecto 0, se declara con la unica
          intencion de utilizar una forma global de llamar a las
          funciones que calculan las soluciones analiticas.
    """
    # Este array define un cero relativo para que la funcion pueda ser calculada
    cero_relativo = np.full_like(vector_valor_eigen, 1e-100, dtype=float)
    # Con np.where se establece lo siguiente: donde hay un valor diferente de cero
    # entonces permanezca dicho valor, si es igual a cero entonces sustituya por cero_relativo
    vector_valor_eigen = np.where(vector_valor_eigen != 0, vector_valor_eigen, cero_relativo)
    # Se crean las variables temporales donde se almacena la operacion evaluada
    # con los desplazamientos en x
    cos_vector_valor_eigen_ca = -(np.cos(vector_valor_eigen * desp_ca) / vector_valor_eigen) * (np.cos(vector_valor_eigen * xf) - np.cos(vector_valor_eigen * xi))
    sin_vector_valor_eigen_ca = (np.sin(vector_valor_eigen * desp_ca) / vector_valor_eigen) * (np.sin(vector_valor_eigen * xf) - np.sin(vector_valor_eigen * xi))
    return (cos_vector_valor_eigen_ca - sin_vector_valor_eigen_ca)


def sol_analitica_cos(xi, xf, vector_valor_eigen, desp_ca, desp_cb=0):
    """
    Funcion que evalua el resultado de forma analitica de la integral:
        cos(vector_valor_eigen*(x-desp_ca)) de xi a xf
    Parametros de Entrada:
        * xi : limite de integracion inicial
        * xf: limite de integracion final
        * vector_valor_eigen: es el valor eigen en la posicion vector_valor_eigen[m]
        * desp_ca: es el desplazamiento en x cuando
        * desp_cb: parametro por defecto 0, se declara con la unica
          intencion de utilizar una forma global de llamar a las
          funciones que calculan las soluciones analiticas.
    """
    # Este array define un cero relativo para que la funcion pueda ser calculada
    cero_relativo = np.full_like(vector_valor_eigen, 1e-100, dtype=float)
    # Con np.where se establece lo siguiente: donde hay un valor diferente de cero
    # entonces permanezca dicho valor, si es igual a cero entonces sustituya por cero_relativo
    vector_valor_eigen = np.where(vector_valor_eigen != 0, vector_valor_eigen, cero_relativo)
    # Se crean las variables temporales donde se almacena la operacion evaluada
    # con los desplazamientos en x
    cos_vector_valor_eigen_ca = (np.cos(vector_valor_eigen * desp_ca) / vector_valor_eigen) * (np.sin(vector_valor_eigen * xf) - np.sin(vector_valor_eigen * xi))
    sin_vector_valor_eigen_ca = (np.sin(vector_valor_eigen * desp_ca) / vector_valor_eigen) * (np.cos(vector_valor_eigen * xf) - np.cos(vector_valor_eigen * xi))
    return (cos_vector_valor_eigen_ca - sin_vector_valor_eigen_ca)


def sol_analitica_por_partes_sin(xi, xf, vector_valor_eigen, desp_ca, desp_cb):
    """
    Funcion que evalua el resultado de forma analitica de la integral:
        cos(vector_valor_eigen*(x-desp_ca)) de xi a xf
    Parametros de Entrada:
        * xi : limite de integracion inicial
        * xf: limite de integracion final
        * vector_valor_eigen: es el valor eigen en la posicion vector_valor_eigen[m]
        * desp_ca: es el
        * desp_cb: parametro por defecto 0, se declara con la unica
          intencion de utilizar una forma global de llamar a las
          funciones que calculan las soluciones analiticas.
    """
    # Este array define un cero relativo para que la funcion pueda ser calculada
    cero_relativo = np.full_like(vector_valor_eigen, 1e-100, dtype=float)
    # Con np.where se establece lo siguiente: donde hay un valor diferente de cero
    # entonces permanezca dicho valor, si es igual a cero entonces sustituya por cero_relativo
    vector_valor_eigen = np.where(vector_valor_eigen != 0, vector_valor_eigen, cero_relativo)
    # Se crean las variables temporales donde se almacena la operacion evaluada
    # con los desplazamientos en x
    k1 = 1 / ((desp_cb - desp_ca) * vector_valor_eigen * vector_valor_eigen) # conocido antes como k3
    k2 = (desp_ca / (desp_cb - desp_ca)) / vector_valor_eigen # conocido antes como k6

    vector_valor_eigen_xi = vector_valor_eigen * xi
    vector_valor_eigen_xf = vector_valor_eigen * xf
    sin_vector_valor_eigen_xf = (k1 * np.sin(vector_valor_eigen_xf)) - (k1 * vector_valor_eigen_xf * np.cos(vector_valor_eigen_xf)) + (k2 * np.cos(vector_valor_eigen_xf ))
    sin_vector_valor_eigen_xi = (k1 * np.sin(vector_valor_eigen_xi)) - (k1 * vector_valor_eigen_xi * np.cos(vector_valor_eigen_xi)) + (k2 * np.cos(vector_valor_eigen_xi))
    return (sin_vector_valor_eigen_xf - sin_vector_valor_eigen_xi)


def sol_analitica_por_partes_cos(xi, xf, vector_valor_eigen, desp_ca, desp_cb):
    """
    Funcion que evalua el resultado de forma analitica de la integral:
        cos(vector_valor_eigen*(x-desp_ca)) de xi a xf
    Parametros de Entrada:
        * xi : limite de integracion inicial
        * xf: limite de integracion final
        * vector_valor_eigen: es el valor eigen en la posicion vector_valor_eigen[m]
        * desp_ca: es el
        * desp_cb: parametro por defecto 0, se declara con la unica
          intencion de utilizar una forma global de llamar a las
          funciones que calculan las soluciones analiticas.
    """
    # Este array define un cero relativo para que la funcion pueda ser calculada
    cero_relativo = np.full_like(vector_valor_eigen, 1e-100, dtype=float)
    # Con np.where se establece lo siguiente: donde hay un valor diferente de cero
    # entonces permanezca dicho valor, si es igual a cero entonces sustituya por cero_relativo
    vector_valor_eigen = np.where(vector_valor_eigen != 0, vector_valor_eigen, cero_relativo)
    # Se crean las variables temporales donde se almacena la operacion evaluada
    # con los desplazamientos en x
    k1 = 1 / ((desp_cb - desp_ca) * vector_valor_eigen * vector_valor_eigen) # conocido antes como k3
    k2 = (desp_ca / ((desp_cb - desp_ca) * vector_valor_eigen)) # conocido antes como k6
    vector_valor_eigen_xi = vector_valor_eigen * xi
    vector_valor_eigen_xf = vector_valor_eigen * xf
    cos_vector_valor_eigen_xf = (k1 * np.cos(vector_valor_eigen_xf)) + (k1 * vector_valor_eigen_xf * np.sin(vector_valor_eigen_xf)) - (k2 * np.sin(vector_valor_eigen_xf ))
    cos_vector_valor_eigen_xi = (k1 * np.cos(vector_valor_eigen_xi)) + (k1 * vector_valor_eigen_xi * np.sin(vector_valor_eigen_xi)) - (k2 * np.sin(vector_valor_eigen_xi))
    return (cos_vector_valor_eigen_xf - cos_vector_valor_eigen_xi)


def cargar_integrandos_vectores_distorsionadores():
    """
    Funcion encargada de cargar en un DataFrame lo necesario para calcular
    los vectores distorsionadores , el archivo vectores_distorsionadores.csv
    debe estar definido en la carpeta csv de la siguiente manera:
        La primera fila se define como la fila de los nombres que tomara cada
    columna, esta fila luce asi:
        xi,xf,tipo,por_partes,ca,cb,chr_eigen

        Donde:
            * xi : limite de integracion inicial
            * xf: limite de integracion final
            * tipo: la funcion trigonometrica involucrada (sin o cos)
            * por_partes: Se indica mediate (si) o (no) si el vector
              distorsionador se integrara por partes, es decir, de la
              siguiente forma:
                  si: ((x-ca)/(cb-ca))*tipo(vector_eigen*x) de xi a xf
                  no: tipo(vector_eigen*x) de xi a xf
            * ca: constante de desplazamiento a
            * cb: constante de desplazamiento b
            * chr_eigen: nombre del vector eigen con el que se integrara.

        Salida:
            * DataFrame integrandos_vectores_distorsionadores: contiene
              la informacion necesaria para calcular los vectores dist.

        Ejemplo de salida:

                xi  xf tipo por_partes  ca  cb    chr_eigen      tipo_f                           calcular_str
            S1   0   6  sin     no       0   0         Q             _sin                   Int( sin(Qm*x) ) de x0 a x6
            S2   0   4  cos     no       0   0         Q             _cos                   Int( cos(Qm*x) ) de x0 a x4
            S3   2   4  cos     si       2   4         Q  _por_partes_cos    Int( (x-x2)/(x4-x2)*cos(Qm*x) ) de x2 a x4
            S4   4   6  sin     no       0   0         Q             _sin                   Int( sin(Qm*x) ) de x4 a x6
            S5   2   4  sin     si       2   4         Q  _por_partes_sin    Int( (x-x2)/(x4-x2)*sin(Qm*x) ) de x2 a x4

    """
    filename = 'csv/' + conf.data['env']['path'] + '/vectores_distorsionadores.csv'
    integrandos_vectores_distorsionadores = pd.read_csv(filename)
    # Se cambian los index de 0,1,2,...ultimo vector distor. a S1,S2,S3,...Sn
    integrandos_vectores_distorsionadores.index = ['S'+str(i) for i in range(1, len(integrandos_vectores_distorsionadores) + 1)]
    # Se agregan el tipo de funcion para luego ser utilizada por la solucion analitica o solucion con integral (_sin, _cos, _por_partes_sin, _por_partes_cos)
    integrandos_vectores_distorsionadores.loc[integrandos_vectores_distorsionadores['por_partes'] == 'no', 'tipo_f'] = '_' + integrandos_vectores_distorsionadores['tipo']
    integrandos_vectores_distorsionadores.loc[integrandos_vectores_distorsionadores['por_partes'] == 'si', 'tipo_f'] = '_por_partes_' + integrandos_vectores_distorsionadores['tipo']
    # Se agrega el calcular string para luego ser usado al momento de plotear
    # Se agrega el calcular string para las integrales sencillas
    integrandos_vectores_distorsionadores.loc[integrandos_vectores_distorsionadores['por_partes'] == 'no', 'calcular_str'] = 'Int( ' + integrandos_vectores_distorsionadores['tipo'] + '('\
                                                                                                                             + integrandos_vectores_distorsionadores['chr_eigen'] + 'm*x) )'\
                                                                                                                             + ' de x' + integrandos_vectores_distorsionadores['xi'].astype(str)\
                                                                                                                             + ' a x' + integrandos_vectores_distorsionadores['xf'].astype(str)
    # Se agrega el calcular string para las integrales por partes
    integrandos_vectores_distorsionadores.loc[integrandos_vectores_distorsionadores['por_partes'] == 'si', 'calcular_str'] = 'Int( (x-x' + integrandos_vectores_distorsionadores['ca'].astype(str) + ')/(x'\
                                                                                                                             + integrandos_vectores_distorsionadores['cb'].astype(str) + '-x'\
                                                                                                                             + integrandos_vectores_distorsionadores['ca'].astype(str) + ')*'\
                                                                                                                             + integrandos_vectores_distorsionadores['tipo'] + '('\
                                                                                                                             + integrandos_vectores_distorsionadores['chr_eigen'] + 'm*x) ) de x'\
                                                                                                                             + integrandos_vectores_distorsionadores['xi'].astype(str)\
                                                                                                                             + ' a x' + integrandos_vectores_distorsionadores['xf'].astype(str)
    return integrandos_vectores_distorsionadores


def calcular_vectores_distorsionadores_integral(integrandos_vectores_distorsionadores, vectores_valores_eigen, n_dimension=100):
    """
    Funcion encargada de calcular los vectores distorsionadores a traves de la
    solucion analitica de todos los casos de integrales.
    Parametros de entrada:
        * integrandos_vectores_distorsionadores: DataFrame que contiene todos los
          valores eigen , tipo de funciones de cada factor del integrando y los
          desplazamiento de cada matriz de acoplamiento.
        * vectores_valores_eigen: DataFrame que contiene todos los vectores
          eigen calculados a partir de los valores eigen.
        * n_dimension: Dimension a utilizar en la creacion de los vectores,
          matrices y demas.

    Salida:
        * vectores_distorsionadores: DataFrame que contiene todos los vectores
          distorsionadores calculados de forma analitica
    """
    m = np.arange(1, n_dimension + 1)
    # Se vectoriza la funcion de integrar para que sea mucho mas rapida su ejecucion
    integrar_func_vectorized = np.vectorize(integrar_func)
    # Se crea el DataFrame de dimension = integrandos_vectores_distorsionadores*n_dimension
    vectores_distorsionadores = pd.DataFrame(np.zeros(((len(integrandos_vectores_distorsionadores.index)), n_dimension)), index=integrandos_vectores_distorsionadores.index)
    # Para evitar confusiones los indices de las columnas van desde 1 en adelante
    vectores_distorsionadores.columns += 1
    print("\n\tCalculando vectores distorsionadores: Solucion con quad\n")
    for Sm in vectores_distorsionadores.index:
        chr_eigen= integrandos_vectores_distorsionadores.loc[Sm, 'chr_eigen']
        tipo_f = integrandos_vectores_distorsionadores.loc[Sm, 'tipo_f']
        xi = integrandos_vectores_distorsionadores.loc[Sm, 'xi']
        xf = integrandos_vectores_distorsionadores.loc[Sm, 'xf']
        desp_ca = integrandos_vectores_distorsionadores.loc[Sm, 'ca']
        desp_cb = integrandos_vectores_distorsionadores.loc[Sm, 'cb']
        vector_valor_eigen = vectores_valores_eigen.loc[chr_eigen].to_numpy()
        print(f"* Vector distorsionador {Sm} = {integrandos_vectores_distorsionadores.loc[Sm, 'calcular_str']}")
        vectores_distorsionadores.loc[Sm] = integrar_func_vectorized('integrando' + tipo_f, xi, xf, vector_valor_eigen, desp_ca, desp_cb)
    return vectores_distorsionadores


def calcular_vectores_distorsionadores_solucion(integrandos_vectores_distorsionadores, vectores_valores_eigen, n_dimension=100):
    """
    Funcion encargada de calcular los vectores distorsionadores a traves de la
    solucion analitica de todos los casos de integrales.
    Parametros de entrada:
        * integrandos_vectores_distorsionadores: DataFrame que contiene todos los
          valores eigen , tipo de funciones de cada factor del integrando y los
          desplazamiento de cada matriz de acoplamiento.
        * vectores_valores_eigen: DataFrame que contiene todos los vectores
          eigen calculados a partir de los valores eigen.
        * n_dimension: Dimension a utilizar en la creacion de los vectores,
          matrices y demas.

    Salida:
        * vectores_distorsionadores: DataFrame que contiene todos los vectores
          distorsionadores calculados de forma analitica
    """
    m = np.arange(1, n_dimension + 1)
    # Se crea el DataFrame de dimension = integrandos_vectores_distorsionadores*n_dimension
    vectores_distorsionadores = pd.DataFrame(np.zeros(((len(integrandos_vectores_distorsionadores.index)), n_dimension)), index=integrandos_vectores_distorsionadores.index)
    # Para evitar confusiones los indices de las columnas van desde 1 en adelante
    vectores_distorsionadores.columns += 1
    print("\n\tCalculando vectores distorsionadores: Solucion analitica\n")
    for Sm in vectores_distorsionadores.index:
        chr_eigen= integrandos_vectores_distorsionadores.loc[Sm, 'chr_eigen']
        tipo_f = integrandos_vectores_distorsionadores.loc[Sm, 'tipo_f']
        xi = integrandos_vectores_distorsionadores.loc[Sm, 'xi']
        xf = integrandos_vectores_distorsionadores.loc[Sm, 'xf']
        desp_ca = integrandos_vectores_distorsionadores.loc[Sm, 'ca']
        desp_cb = integrandos_vectores_distorsionadores.loc[Sm, 'cb']
        vector_valor_eigen = vectores_valores_eigen.loc[chr_eigen].to_numpy()
        print(f"* Vector distorsionador {Sm} = {integrandos_vectores_distorsionadores.loc[Sm, 'calcular_str']}")
        vectores_distorsionadores.loc[Sm] = eval('sol_analitica' + tipo_f + '(xi, xf, vector_valor_eigen, desp_ca, desp_cb)')
    return vectores_distorsionadores
