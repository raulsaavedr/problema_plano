from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

import numpy as np
import pandas as pd

from config import conf
import eigen as eig
import region as reg
import hiperbolica as hyp
import matrices_acoplamiento as m_acop
import distorsionador as v_dist
import matriz_gauss as m_gauss
import v_transpuestos as v_trans
import potencial as pot
import flujo as flj

__doc__ = """
Este modulo esta hecho para graficar los valores con error y sin error de los
diferentes componentes del calculo del problema plano. Es equivalente a todas
las funciones de prueba.
"""

def prueba_valor_eigen(valores_eigen, vectores_valores_eigen, vectores_valores_eigen_err,\
                       n_dimension=100, error=1):
    """
    Funcion encargada de graficar con matplotlib los vectores eigen versus los
    vectores eigen calculados con error.Equivalente a : f03_Valor_Eigen_Prueba()
    Pametros de entrada:
        * valores_eigen : DataFrame encargado de guardar los valores eigen
          representativos de todas las regiones.
        * vectores_valores_eigen: DataFrame que almacena los valores calculados
          de cada vector eigen. Es decir Pn = [Veig[0],..., Veig[n_dimension]] ,
          Qn = [Veig[0],..., Veig[n_dimension]] y así conlos demas vectores.

    Salida:
        * Guarda las figuras en la carpeta ../graficas/vectores eigen . Esta car
          peta debe estar previamente creada para que no haya conflictos al mo-
          mento de guardar las graficas.

    Nota: Para un ejemplo dado remitirse a la funcion main de graficacion.py.
    """
    error_t = 'sin error ' if error == 1 else 'con error'
    N = range(1, n_dimension + 1)        # Solo para propositos de graficacion
    for chr_eigen in valores_eigen.index:
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        # Encuentre el maximo de valor eigen entre el comparado y el original (para efectos de graficacion)
        maximo = np.max((vectores_valores_eigen.loc[chr_eigen].max(), vectores_valores_eigen_err.loc[chr_eigen].max()))
        fig.suptitle(f"Control {error_t} nr={n_dimension} del Valor Eigen {chr_eigen+'='+valores_eigen.loc[chr_eigen]['calcular_str']}")
        ax.text(n_dimension / 8, 0.95 * maximo, """Prueba correcta si se imprime una sola curva.
                 Error si imprime dos curvas""")
        # Grafique vector transpuesto de Valores eigen de la funcion (sin error) Color rojo
        ax.plot(N, vectores_valores_eigen.loc[chr_eigen], 'r', label='sin error')
        # Grafique vector transpuesto de Valores eigen de la ecuacion (con error) Color azul
        ax.plot(N, vectores_valores_eigen_err.loc[chr_eigen], 'b', label='con error')
        ax.legend(loc='lower right')
        filename = 'graficas/' + conf.data['env']['path'] + '/vectores eigen/control ' +\
                    error_t + " " + chr_eigen + ".svg"
        canvas = FigureCanvas(fig)
        canvas.print_figure(filename)
        # Salida por consola del proceso que se esta realizando
        print(f"* {chr_eigen+'='+valores_eigen.loc[chr_eigen]['calcular_str']}")


def prueba_matrices_diagonal_valores_eigen(valores_eigen, vectores_valores_eigen,\
                                           vectores_valores_eigen_err_matriz, n_dimension=100, error=1):
    """
    Funcion encargada de graficar con matplotlib las matrices diagonales de va-
    lores eigen versus las matrices de valores eigen calculados con error.
    Equivalente a: f04_Diag_Valor_Eigen_Prueba()
    Pametros de entrada:
        * valores_eigen : DataFrame encargado de guardar los valores eigen
          representativos de todas las regiones.
        * vectores_valores_eigen: DataFrame que almacena los valores calculados
          de cada vector eigen. Es decir Pn = [Veig[0],..., Veig[n_dimension]] ,
          Qn = [Veig[0],..., Veig[n_dimension]] y así conlos demas vectores.
        * vectores_valores_eigen_err_matriz: Dataframe en donde esta almacenado
          la matriz diagonal de los valores eigen calculados con un error dado.

    Salida:
        * Guarda las figuras en la carpeta ../graficas/matrices diagonales de
          valores eigen . Esta carpeta debe estar previamente creada para que no
          haya conflictos al momento de guardar las graficas.
    """
    error_t = 'sin error ' if error == 1 else 'con error'
    N = range(1, n_dimension + 1)       # Solo para propositos de graficacion
    for chr_eigen in vectores_valores_eigen.index:
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        # Encuentre el maximo de valor eigen entre el comparado y el original (para efectos de graficacion del texto)
        maximo = np.max((vectores_valores_eigen.loc[chr_eigen].max(), vectores_valores_eigen_err_matriz.loc[chr_eigen].max()))
        fig.suptitle(f"{error_t.capitalize()} - nr={n_dimension} - Matriz diagonal del valor Eigen: {chr_eigen+'='+valores_eigen.loc[chr_eigen]['calcular_str']}")
        ax.text(n_dimension / 8, 0.95 * maximo, """Prueba correcta si se imprime una sola curva.
                 Error si imprime dos curvas""")
        ax.plot(N, vectores_valores_eigen.loc[chr_eigen], 'r', label='sin error')
        ax.plot(N, vectores_valores_eigen_err_matriz.loc[chr_eigen], 'b', label='con error')
        ax.legend(loc='lower right')
        filename = 'graficas/' + conf.data['env']['path'] + '/matrices diagonales de valores eigen/' +\
                    'control ' + error_t + " " + chr_eigen + ".svg"
        canvas = FigureCanvas(fig)
        canvas.print_figure(filename)
        print(f"* Matriz diagonal {chr_eigen+'='+valores_eigen.loc[chr_eigen]['calcular_str']}")


def prueba_matrices_diagonal_funciones_hiperbolicas(funciones_hiperbolicas, vectores_funciones_hiperbolicas,\
                                                    vectores_funciones_hiperbolicas_err, n_dimension=100, error=1):
    """
    Funcion encargada de graficar con matplotlib las matrices diagonal de las
    funciones hiperbólicas eigen versus las matrices de valores eigen calculados
    con error. Equivalente a: f05_Diag_Func_Hiper_Prueba()
    Pametros de entrada:
        * funciones_hiperbolicas: Es el DataFrame creado en donde estan almacena
          dos todos los valores necesarios para poder calcular los vectores de
          funciones hiperbolicas.
        * vectores_funciones_hiperbolicas: Es un DataFrame que contiene todos
          los valores calculados de las funciones hiperbolicas  de todos los
          vectores.
        * vectores_funciones_hiperbolicas_err: Es un DataFrame que contiene todos
          los valores calculados dado un error de las funciones hiperbolicas  de
          todos los vectores.
    Salida:
        * Guarda las figuras en la carpeta "../graficas/matrices diagonales de
         funciones hiperbolicas". Esta carpeta debe estar previamente creada
         para que no haya conflictos al momento de guardar las graficas.
    """
    error_t = 'sin error ' if error == 1 else 'con error'
    N = range(1, n_dimension + 1)  # Solo para propositos de graficacion
    # Se obtiene un index a partir de la matrices diagonales
    for nro_diagonal in funciones_hiperbolicas.index:
        # Encuentre el maximo de valor eigen entre el comparado y el original (para efectos de graficacion)
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        maximo = np.max((vectores_funciones_hiperbolicas.loc[nro_diagonal].max(), vectores_funciones_hiperbolicas_err.loc[nro_diagonal].max()))
        minimo = np.min((vectores_funciones_hiperbolicas.loc[nro_diagonal].min(), vectores_funciones_hiperbolicas_err.loc[nro_diagonal].min()))
        fig.suptitle(f"{error_t.capitalize()} - nr={n_dimension} - Control de la matriz diagonal: {nro_diagonal+'='+funciones_hiperbolicas.loc[nro_diagonal]['calcular_str']}")
        ax.text( 0.1 * n_dimension, minimo + ((maximo - minimo) / 2), """Prueba correcta si se imprime una sola curva.
                 Error si imprime dos curvas""")
        # plt.axvline(0.1 * n_dimension, color='k', linestyle='solid')
        # plt.axhline(.00005 * maximo, color='k', linestyle='solid')
        ax.plot(N, vectores_funciones_hiperbolicas.loc[nro_diagonal], 'r', label='sin error')
        ax.plot(N, vectores_funciones_hiperbolicas_err.loc[nro_diagonal], 'b', label='con error')
        ax.legend(loc='lower right')
        filename = 'graficas/' + conf.data['env']['path'] + '/matrices diagonales de funciones hiperbolicas/' +\
                   'control ' + error_t + " " + nro_diagonal + ".svg"
        canvas = FigureCanvas(fig)
        canvas.print_figure(filename)
        print(f"* Matriz diagonal {nro_diagonal+'='+funciones_hiperbolicas.loc[nro_diagonal]['calcular_str']}")


def prueba_matrices_cuadradas_acoplamiento(integrandos_matrices_acoplamiento, matrices_acoplamiento_int,\
                                           matrices_acoplamiento_sol, n_dimension=100, error=1):
    """
    Funcion encargada de graficar con matplotlib las matrices cuadradas de aco-
    acoplamiento solucion analitica versus solucion por quad de scipy.
    Equivalente a: f05_Diag_Func_Hiper_Prueba()
    Pametros de entrada:

    Salida:
        * Guarda las figuras en la carpeta "../graficas/matrices cuadradas de
          acoplamiento". Esta carpeta debe estar previamente creada
          para que no haya conflictos al momento de guardar las graficas.
    """
    error_t = 'sin error ' if error == 1 else 'con error'
    N = range(1, (n_dimension * n_dimension) + 1)        # Solo para propositos de graficacion
    # Se obtiene un index a partir del df integrandos_matrices_acoplamiento
    matrices_acoplamiento_sol = error * matrices_acoplamiento_sol
    for M in integrandos_matrices_acoplamiento.index:
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        # Encuentre el maximo de valor de las dos matrices (la matriz sin error y la matriz con error) (para efectos de graficacion)
        # Nota importante: A cada matriz se le hace un stack durante todo el proceso para obtener un vector de todos los valores de la Matriz
        maximo = np.max((matrices_acoplamiento_int.loc[M].stack().loc[:n_dimension,:n_dimension].max(), matrices_acoplamiento_sol.loc[M].stack().loc[:n_dimension,:n_dimension].max()))
        fig.suptitle(f"{error_t.capitalize()} - nr={n_dimension} - {M + '=' + integrandos_matrices_acoplamiento.loc[M, 'calcular_str']}")
        # ax.text( 0.5 * (n_dimension ** 2), maximo, """Prueba correcta si se imprime una sola grafica.
                 # Error si imprime dos graficas""")
        ax.plot(N, matrices_acoplamiento_int.loc[M].stack().loc[:n_dimension,:n_dimension], 'r', label='sol. integrate.quad')
        ax.plot(N, matrices_acoplamiento_sol.loc[M].stack().loc[:n_dimension,:n_dimension], 'b', label='sol. analitica ' + error_t)
        ax.legend(loc='lower right')
        filename = 'graficas/' + conf.data['env']['path'] + '/matrices cuadradas de acoplamiento/' +\
                   'control ' + error_t + " " + M + ".svg"
        canvas = FigureCanvas(fig)
        canvas.print_figure(filename)
        print(f"* Matriz acompladora {M + '=' + integrandos_matrices_acoplamiento.loc[M, 'calcular_str']}")



def prueba_vectores_distorsionadores(integrandos_vectores_distorsionadores, vectores_distorsionadores_int,\
                                     vectores_distorsionadores_sol, n_dimension=100, error=1):
    """
    Funcion encargada de graficar con matplotlib los vectores distorsionadores
    versus solucion por quad de scipy.
    Equivalente a: f05_Diag_Func_Hiper_Prueba()
    Pametros de entrada:

    Salida:
        * Guarda las figuras en la carpeta "../graficas/matrices cuadradas de
          acoplamiento". Esta carpeta debe estar previamente creada
          para que no haya conflictos al momento de guardar las graficas.
    """
    error_t = 'sin error ' if error == 1 else 'con error'
    N = range(1, n_dimension + 1)        # Solo para propositos de graficacion
    # Se agrega un error a la solucion analitica
    vectores_distorsionadores_sol = error * vectores_distorsionadores_sol
    # Se obtiene un index a partir del df integrandos_vectores_distorsionadores
    for Sm in integrandos_vectores_distorsionadores.index:
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        # Encuentre el maximo de valor de las dos matrices (la matriz sin error y la matriz con error) (para efectos de graficacion)
        maximo = np.max((vectores_distorsionadores_int.loc[Sm][:n_dimension].max(), vectores_distorsionadores_sol.loc[Sm][:n_dimension].max()))
        plt.xticks(N)
        fig.suptitle(f"{error_t.capitalize()} - nr={n_dimension} - Vector Dist.: {Sm + '=' + integrandos_vectores_distorsionadores.loc[Sm, 'calcular_str']}")
        # ax.text( 0.5 * (n_dimension ** 2), maximo, """Prueba correcta si se imprime una sola grafica.
                 # Error si imprime dos graficas""")
        ax.plot(N, vectores_distorsionadores_int.loc[Sm][:n_dimension], 'r', label='sol. integrate.quad')
        ax.plot(N, vectores_distorsionadores_sol.loc[Sm][:n_dimension], 'b', label='sol. analitica '+error_t)
        ax.legend(loc='upper right')
        filename = 'graficas/' + conf.data['env']['path'] + '/vectores distorsionadores/' +\
                   'control ' + error_t + " " + Sm + ".svg"
        canvas = FigureCanvas(fig)
        canvas.print_figure(filename)
        print(f"* Vector distorsionador {Sm + '=' + integrandos_vectores_distorsionadores.loc[Sm, 'calcular_str']}")


def prueba_potencial(regiones, recursos_potencial, potenciales, potenciales_err, dimension_mesh,\
                     n_dimension=100, error =1):
    """
    Funcion encargada de graficar con matplotlib los vectores eigen versus los
    vectores eigen calculados con error.Equivalente a : f12_V_dV_Prueba()
    Pametros de entrada:

    Salida:
        * Guarda las figuras en la carpeta ../graficas/potenciales . Esta car-
          peta debe estar previamente creada para que no haya conflictos al mo-
          mento de guardar las graficas.
    """
    error_t = 'sin error ' if error == 1 else 'con error'
    N = range(1, dimension_mesh + 1)        # Solo para propositos de graficacion
    for n_potencial in potenciales.index:
        # Reg.1, Reg.2, ... Reg.n - Iteradores de las regiones
        index_reg_actual = "Reg." + n_potencial.split('V')[1]
        # Encuentre el maximo de valor eigen entre el comparado y el original (para efectos de graficacion)
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        maximo = np.max((potenciales.loc[n_potencial].max(), potenciales_err.loc[n_potencial].max()))
        minimo = np.max((potenciales.loc[n_potencial].min(), potenciales_err.loc[n_potencial].min()))
        fig.suptitle(f"Con nr={n_dimension}- Prueba del potencial {error_t} de la {index_reg_actual}-{regiones.loc[index_reg_actual, 'eps']}.")
        ax.text(n_dimension / 8, 0.95 * maximo, """Prueba correcta si se imprime una sola curva.
                 Error si imprime dos curvas""")
        # Grafique potenciales (con error) Color rojo
        ax.plot(N, potenciales_err.loc[n_potencial], 'r', label='con error')
        # Grafique potenciales (sin error) Color negro
        ax.plot(N, potenciales.loc[n_potencial], 'k', label='sin error')
        ax.legend(loc='lower right')
        filename = 'graficas/' + conf.data['env']['path'] + '/potenciales/' +\
                   'control ' + error_t + " " + n_potencial + ".svg"
        canvas = FigureCanvas(fig)
        canvas.print_figure(filename)
        # Salida por consola del proceso que se esta realizando
        print(f"* {n_potencial}={recursos_potencial.loc[n_potencial,'calcular_str']}")

def prueba_flujo(regiones, recursos_flujo, flujos, flujos_err, dimension_mesh,\
                 n_dimension=100, error=1):
    """
    Funcion encargada de graficar con matplotlib los vectores eigen versus los
    vectores eigen calculados con error.Equivalente a : f12_V_dV_Prueba()
    Pametros de entrada:

    Salida:
        * Guarda las figuras en la carpeta ../graficas/flujos. Esta carpeta
          debe estar previamente creada para que no haya conflictos al momento
          de guardar las graficas.
    """
    error_t = 'sin error ' if error == 1 else 'con error'
    N = range(1, dimension_mesh + 1)        # Solo para propositos de graficacion
    for n_flujo in flujos.index:
        # Reg.1, Reg.2, ... Reg.n - Iteradores de las regiones
        index_reg_actual = "Reg." + n_flujo.split('V')[1]
        # Encuentre el maximo de valor eigen entre el comparado y el original (para efectos de graficacion)
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        maximo = np.max((flujos.loc[n_flujo].max(), flujos_err.loc[n_flujo].max()))
        minimo = np.max((flujos.loc[n_flujo].min(), flujos_err.loc[n_flujo].min()))
        fig.suptitle(f"Con nr={n_dimension}- Prueba del flujo {error_t} de la {index_reg_actual}-{regiones.loc[index_reg_actual, 'eps']}.")
        # ax.text(n_dimension / 8, 0.95 * maximo, """Prueba correcta si se imprime una sola curva.
        #          Error si imprime dos curvas""")
        # Grafique flujos (con error) Color rojo
        ax.plot(N, flujos_err.loc[n_flujo], 'r', label='con error')
        # Grafique flujos (sin error) Color negro
        ax.plot(N, flujos.loc[n_flujo], 'k', label='sin error')
        ax.legend(loc='lower right')
        filename = 'graficas/' + conf.data['env']['path'] + '/flujos/' +\
                   'control ' + error_t + " " + n_flujo + ".svg"
        canvas = FigureCanvas(fig)
        canvas.print_figure(filename)
        # Salida por consola del proceso que se esta realizando
        print(f"* {n_flujo}={recursos_flujo.loc[n_flujo,'calcular_str']}")

def graficas_potencial(regiones, potenciales, mesh_regiones, n_dimension):

    for n_potencial in potenciales.index:
        index_reg_actual = "Reg." + n_potencial.split('V')[1]
        pot = potenciales.loc[n_potencial].to_numpy()
        pot = np.reshape(pot, (int(np.sqrt(len(pot))),int(np.sqrt(len(pot)))))
        x_flat = mesh_regiones.loc[index_reg_actual,'x'].to_numpy()
        x_flat = np.reshape(x_flat, (int(np.sqrt(len(x_flat))),int(np.sqrt(len(x_flat)))))
        y_flat = mesh_regiones.loc[index_reg_actual,'y'].to_numpy()
        y_flat = np.reshape(y_flat, (int(np.sqrt(len(y_flat))),int(np.sqrt(len(y_flat)))))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle(f"Con nr={n_dimension}- surf del potencial de la {index_reg_actual}-{regiones.loc[index_reg_actual, 'eps']}.")
        ax.plot_surface(x_flat,y_flat,pot,cmap=cm.autumn)
        #ax.view_init(0,-90)
        filename = 'graficas/' + conf.data['env']['path'] + '/potenciales/surf/' +\
                   'Surf' + " " + n_potencial + ".svg"
        canvas = FigureCanvas(fig)
        canvas.print_figure(filename)
        print('.', end='')        
