import region as reg
import eigen as eig
import hiperbolica as hyp
import numpy as np
import pandas as pd
from scipy.integrate import quad

"""
La funciones integrando son el integrando de la integral valga la redundancia
dichas funciones reciben como parametro implicito 'x' que es el que utiliza la
funcion quad para integrar
"""
def integrando_sin_sin(x, veig1, d1, veig2, d2):
    return np.sin(veig1*(x-d1))*np.sin(veig2*(x-d2))


def integrando_sin_cos(x, veig1, d1, veig2, d2):
    return np.sin(veig1*(x-d1))*np.cos(veig2*(x-d2))


def integrando_cos_sin(x, veig1, d1, veig2, d2):
    return np.cos(veig1*(x-d1))*np.sin(veig2*(x-d2))


def integrando_cos_cos(x, veig1, d1, veig2, d2):
    return np.cos(veig1*(x-d1))*np.cos(veig2*(x-d2))


def integrar_func(func_str, xi, xf, veig1, d1, veig2, d2):
    """
    Paramentros de entrada:
        * func_str : es la funcion a ser llamada y se almacena en el DataFrame
          integrandos_matrices_acoplamiento['tipo_f'], esta cadena luego sera
          evaluada por la funcion de python eval() para ser pasada como parametro
          a la funcion quad
        * xi : limite de integracion inicial
        * xf: limite de integracion final
        * veig1: es el valor eigen en la posicion veig1[m]
        * d1: desplazamiento en x de la primera funcion
        * veig1: es el valor eigen en la posicion veig2[n]
        * d2: desplazamiento en x de la segunda funcion
    """
    return quad(eval(func_str), xi, xf, args=(veig1, d1, veig2, d2), limit=100000)[0]


"""
Estas funciones son las funciones ya integradas de las matrices de acomplamiento
y se evaluan con el teorema fundamental del calculo. A cada resta que haya en el
divisor se hara un manejo del error division por cero con la funcion np.divide()
se divide el dividendo que es a_xf o a_xi entre el divisor que es el delta invo-
lucrado. Si arroja un valor Nan o un error de division por cero reemplazara ese
valor por cero.
Parametros de Entrada:
    * xi : limite de integracion inicial
    * xf: limite de integracion final
    * veig1: es el valor eigen en la posicion veig1[m]
    * d1: desplazamiento en x de la primera funcion
    * veig1: es el valor eigen en la posicion veig2[n]
    * d2: desplazamiento en x de la segunda funcion
"""
def sin_sin(xi, xf, veig1, d1, veig2, d2):
    a_xf = np.sin((d1 * veig1 - d2 * veig2) - (veig1 - veig2) * xf)
    a_xi = np.sin((d1 * veig1 - d2 * veig2) - (veig1 - veig2) * xi)
    delta = (2 * (veig1 - veig2))
    return (np.sin((d1 * veig1 + d2 * veig2) - (veig1 + veig2) * xf) / (2 * (veig1 + veig2))\
            - np.divide(a_xf, delta, out=np.zeros(a_xf.shape, dtype=float), where=delta!=0))\
            -(np.sin((d1 * veig1 + d2 * veig2) - (veig1 + veig2) * xi) / (2 * (veig1 + veig2))\
                    -  np.divide(a_xi, delta, out=np.zeros(a_xi.shape, dtype=float), where=delta!=0))


def sin_cos(xi, xf, veig1, d1, veig2, d2):
    a_xf = np.cos((d1 * veig1 - d2 * veig2) + (veig2 - veig1) * xf)
    a_xi = np.cos((d1 * veig1 - d2 * veig2) + (veig2 - veig1) * xi)
    delta = (2 * (veig2 - veig1))
    return (np.divide(a_xf, delta, out=np.zeros(a_xf.shape, dtype=float), where=delta!=0)\
            - np.cos((d1 * veig1 + d2 * veig2) - (veig1 + veig2) * xf) / (2 * (veig1 + veig2)))\
            -(np.divide(a_xi, delta, out=np.zeros(a_xi.shape, dtype=float), where=delta!=0)\
                    - np.cos((d1 * veig1 + d2 * veig2) - (veig1 + veig2) * xi) / (2 * (veig1 + veig2)))


def cos_sin(xi, xf, veig1, d1, veig2, d2):
    a_xf = np.cos((d1 * veig1 - d2 * veig2) + (veig2 - veig1) * xf)
    a_xi = np.cos((d1 * veig1 - d2 * veig2) + (veig2 - veig1) * xi)
    delta = (2 * (veig1 - veig2))
    return (np.divide(a_xf, delta, out=np.zeros(a_xf.shape, dtype=float), where=delta!=0)\
            - np.cos((d1 * veig1 + d2 * veig2) - (veig1 + veig2) * xf) / (2 * (veig1 + veig2)))\
            -(np.divide(a_xi, delta, out=np.zeros(a_xi.shape, dtype=float), where=delta!=0)\
                    - np.cos((d1 * veig1 + d2 * veig2) - (veig1 + veig2) * xi) / (2 * (veig1 + veig2)))


def cos_cos(xi, xf, veig1, d1, veig2, d2):
    a_xf = np.sin((d1 * veig1 - d2 * veig2) + (veig2 - veig1) * xf)
    a_xi = np.sin((d1 * veig1 - d2 * veig2) + (veig2 - veig1) * xi)
    delta = (2 * (veig2 - veig1))
    return (np.divide(a_xf, delta, out=np.zeros(a_xf.shape, dtype=float), where=delta!=0)\
            - np.sin((d1 * veig1 + d2 * veig2) - (veig1 + veig2) * xf) / (2 * (veig1 + veig2)))\
            -(np.divide(a_xi, delta, out=np.zeros(a_xi.shape, dtype=float), where=delta!=0)\
                    - np.sin((d1 * veig1 + d2 * veig2) - (veig1 + veig2) * xi) / (2 * (veig1 + veig2)))


def cargarIntegrandosMatricesAcoplamiento():
    """
    Se carga en un dataframe lo necesario para contruir como se calcularan
    las matrices cuadradas de acomplamiento.
    """
    integrandos_matrices_acoplamiento = pd.read_csv('csv/matrices_cuadradas_acomplamiento.csv')
    # Se cambian los index de 0,1,2,...ultima matriz a M1,M2,M3,...Mn
    integrandos_matrices_acoplamiento.index = ['M'+str(i) for i in range(1, len(integrandos_matrices_acoplamiento.index) + 1)]
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
    integrandos_matrices_acoplamiento
    return integrandos_matrices_acoplamiento


def calcularMatricesAcomplamientoIntegral(integrandos_matrices_acoplamiento, vectores_valores_eigen, n_dimension=100):
    """
    """
    # Se vectoriza la funcion de integrar para que sea mucho mas rapida su ejecucion
    integrar_func_vectorized = np.vectorize(integrar_func)
    Mn=np.empty((8,n_dimension,n_dimension), dtype=float)
    print("\n\t\t\tSolucion con Scipy quad")
    for M in integrandos_matrices_acoplamiento.index:
        chr_eigen_1 = integrandos_matrices_acoplamiento.loc[M, 'chr_eigen_1']
        chr_eigen_2 = integrandos_matrices_acoplamiento.loc[M, 'chr_eigen_2']
        tipo_f = integrandos_matrices_acoplamiento.loc[M,'tipo_f']
        xi = integrandos_matrices_acoplamiento.loc[M, 'xi']
        xf = integrandos_matrices_acoplamiento.loc[M, 'xf']
        d1 = integrandos_matrices_acoplamiento.loc[M, 'desp_1']
        d2 = integrandos_matrices_acoplamiento.loc[M, 'desp_2']
        veig1 = vectores_valores_eigen.loc[chr_eigen_1].to_numpy()
        veig2 = vectores_valores_eigen.loc[chr_eigen_2].to_numpy()[:,np.newaxis]
        i = int(M.split('M')[1])-1
        print(f"\nCalculando:\n* Matriz {M} = {integrandos_matrices_acoplamiento.loc[M, 'calcular_str']}\n")
        Mn[i] = integrar_func_vectorized('integrando_' + tipo_f,xi,xf,veig1, d1,veig2, d2).T
        print(Mn[i])
    return Mn

def calcularMatricesAcomplamientoSolucion(integrandos_matrices_acoplamiento, vectores_valores_eigen, n_dimension=100):
    """
    """
    Mn=np.empty((8,n_dimension,n_dimension), dtype=float)
    print("\n\t\t\t\tSolucion Analitica")
    for M in integrandos_matrices_acoplamiento.index:
        chr_eigen_1 = integrandos_matrices_acoplamiento.loc[M, 'chr_eigen_1']
        chr_eigen_2 = integrandos_matrices_acoplamiento.loc[M, 'chr_eigen_2']
        tipo_f = integrandos_matrices_acoplamiento.loc[M,'tipo_f']
        xi = integrandos_matrices_acoplamiento.loc[M, 'xi']
        xf = integrandos_matrices_acoplamiento.loc[M, 'xf']
        d1 = integrandos_matrices_acoplamiento.loc[M, 'desp_1']
        d2 = integrandos_matrices_acoplamiento.loc[M, 'desp_2']
        veig1 = vectores_valores_eigen.loc[chr_eigen_1].to_numpy()
        veig2 = vectores_valores_eigen.loc[chr_eigen_2].to_numpy()[:,np.newaxis]
        i = int(M.split('M')[1])-1
        print(f"\nCalculando:\n* Matriz {M} = {integrandos_matrices_acoplamiento.loc[M, 'calcular_str']}\n")
        Mn[i] = eval(tipo_f + '(xi, xf, veig1, d1, veig2, d2)').T
        print(Mn[i])
    return Mn



if __name__=='__main__':
    regiones = reg.cargarRegiones(5)  # region.cargarRegiones(2000000)
    # Carga de dimension a partir de las regiones
    n_dimension = regiones['n_dimension'][0]
    # Carga de los valores eigen significativos de las regiones
    valores_eigen = eig.cargarValoresEigen(regiones)
    # Calculo de los vectores de los valores eigen
    vectores_valores_eigen = eig.calcularVectoresValoresEigen(valores_eigen, n_dimension)
    integrandos_matrices_acoplamiento = cargarIntegrandosMatricesAcoplamiento()
    Mn_quad = calcularMatricesAcomplamientoIntegral(integrandos_matrices_acoplamiento, vectores_valores_eigen, n_dimension)
    Mn_sol = calcularMatricesAcomplamientoSolucion(integrandos_matrices_acoplamiento, vectores_valores_eigen, n_dimension)
    for M in integrandos_matrices_acoplamiento.index:
        i = int(M.split('M')[1])-1
        print('\nMn sol[i][0]')
        print(Mn_sol[i][0])
        print('Mn quad[i][0]')
        print(Mn_quad[i][0])
        print("Diferencia vector")
        print(Mn_sol[i][0]-Mn_quad[i][0])
        print("Diferencia matriz")
        print(Mn_sol[i] - Mn_quad[i])
