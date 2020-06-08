from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

from matplotlib.patches import Rectangle, PathPatch
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
import mpl_toolkits.mplot3d.art3d as art3d

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

def control_de_continuidad(regiones, potenciales, mesh_regiones, n_dimension):
    continuidad = pd.read_csv('csv/' + conf.data['env']['path'] + '/continuidad.csv')

    for index in continuidad.index:
        fig = plt.figure()

        R_sup = continuidad.loc[index,'region_superior'].split('R')[1]
        R_inf = continuidad.loc[index,'region_inferior'].split('R')[1]

        X_sup = mesh_regiones.loc['Reg.'+R_sup,'x'].to_numpy()
        X_sup = np.reshape(X_sup, (int(np.sqrt(len(X_sup))),int(np.sqrt(len(X_sup)))))[0]

        X_inf = mesh_regiones.loc['Reg.'+R_inf,'x'].to_numpy()
        X_inf = np.reshape(X_inf, (int(np.sqrt(len(X_inf))),int(np.sqrt(len(X_inf)))))[0]

        pot_superior = potenciales.loc['V'+R_sup].to_numpy()
        pot_superior = np.reshape(pot_superior , (int(np.sqrt(len(pot_superior))),int(np.sqrt(len(pot_superior)))))[0]

        pot_inferior = potenciales.loc['V'+R_inf].to_numpy()
        pot_inferior = np.reshape(pot_inferior, (int(np.sqrt(len(pot_inferior))),int(np.sqrt(len(pot_inferior)))))[-1]

        left_bar  = [continuidad.loc[index,'xi'],continuidad.loc[index,'xi']]
        right_bar = [continuidad.loc[index,'xf'],continuidad.loc[index,'xf']]

        plt.title(f"Con nr={n_dimension}- Prueba de continuidad potencial de la Reg.{R_inf} a la Reg.{R_sup}")

        plt.plot(X_sup, pot_superior, 'r')
        plt.plot(X_inf, pot_inferior, 'b')

        #ESTO SON LOS PUNTOS DONDE DEBEN COINCIDIR LAS GRAFICAS
        plt.plot(left_bar, [-2,2])
        plt.plot(right_bar, [-2,2])

        filename ='graficas/' + conf.data['env']['path'] + '/continuidad de potencial/'+ f'Reg.{R_inf} a la Reg.{R_sup}.svg'
        canvas = FigureCanvas(fig)
        canvas.print_figure(filename)
        plt.close()

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
        plt.close()
        print('.', end='')
    print()


def grafica_de_potencial_total(regiones, potenciales, mesh_regiones, n_dimension):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title(f"Grafica de los niveles de potencial de todas las regiones")

    for n_potencial in potenciales.index:

        index_reg_actual = "Reg." + n_potencial.split('V')[1]

        pot = potenciales.loc[n_potencial].to_numpy()
        pot = np.reshape(pot, (int(np.sqrt(len(pot))),int(np.sqrt(len(pot)))))
        x_flat = mesh_regiones.loc[index_reg_actual,'x'].to_numpy()
        x_flat = np.reshape(x_flat, (int(np.sqrt(len(x_flat))),int(np.sqrt(len(x_flat)))))
        y_flat = mesh_regiones.loc[index_reg_actual,'y'].to_numpy()
        y_flat = np.reshape(y_flat, (int(np.sqrt(len(y_flat))),int(np.sqrt(len(y_flat)))))

        ax.plot_surface(x_flat,y_flat,pot,cmap=cm.autumn)

    ax.view_init(0,-90)
    filename ='graficas/' + conf.data['env']['path'] + '/Grafica de Potencial total.svg'
    canvas = FigureCanvas(fig)
    canvas.print_figure(filename)
    plt.close()

def draw_rectangle(ax, inicio= 0, ancho= 2,direction = 'y',desp= 2, alto= 3, fill= True):
    rect = Rectangle((inicio,0), width= ancho, height=alto, fill= fill)
    if not fill:
        rect.set_edgecolor('r')
    else:
        rect.set_edgecolor('k')
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=desp, zdir=direction)

def draw_text(ax, x, y, z=1, cadena=''):

    text_path = TextPath((0, 0), cadena, size=.35)
    trans = Affine2D().translate(x, y)

    t1 = PathPatch(trans.transform_path(text_path), fc='k')
    ax.add_patch(t1)
    art3d.pathpatch_2d_to_3d(t1, z=z, zdir='z')

def draw_region3d(ax, xi, xf, yi, yf, fronteras, n_region, material, z=1, xmax=None):

    #texto
    x_texto = xi if xi<xf else xf
    desp_t = abs(xf-xi)*.2 if abs(xf-xi)==1 else abs(xf-xi)*.4
    draw_text(ax,x_texto+desp_t,yi + (yf-yi)*.3,z,f'R{n_region} {material}')

    for lugar, valor in fronteras.items():

        if lugar=='arriba':
            x_t = xi+(xf-xi)/2
            y_t = yf
        elif lugar=='abajo':
            x_t = xi+(xf-xi)/2
            y_t = yi
        elif lugar=='derecha':
            x_t = xf
            y_t = yi+(yf-yi)/2
        elif lugar=='izquierda':
            x_t = xi
            y_t = yi+(yf-yi)/2

        if   valor == 'Uno':  texto = f'V{n_region}=1'
        elif valor == 'Cero': texto = f'V{n_region}=0'
        elif valor == 'SIM':  texto = 'SIM'

        if valor in ['Uno','Cero','SIM']: draw_text(ax,x_t,y_t,z,texto)

        direccion =    'y' if lugar=='arriba' or lugar=='abajo' else 'x'
        punto_inicial = xi if lugar=='arriba' or lugar=='abajo' else yi
        ancho =    (xf-xi) if lugar=='arriba' or lugar=='abajo' else (yf-yi)

        if   lugar=='arriba':    desp=yf
        elif lugar=='abajo':     desp=yi
        elif lugar=='derecha':   desp=xf
        elif lugar=='izquierda': desp=xi

        if valor == 'Uno' or valor == 'Cero':
            draw_rectangle(ax, inicio= punto_inicial, ancho= ancho, direction= direccion, desp= desp, alto= z)
        elif valor=='no' or valor=='SIM':
            draw_rectangle(ax, inicio= punto_inicial, ancho= ancho, direction= direccion, desp= desp, alto= z,fill=False)
        else:
        #Fronteras Compuestas:
            front_list = [x.split('-') for x in valor.split('/')]
            for pseudo_frontera in front_list:
                punto_inicial = int(pseudo_frontera[1])
                ancho =    (int(pseudo_frontera[2])-int(pseudo_frontera[1]))

                if pseudo_frontera[0] == 'Uno' or pseudo_frontera[0] == 'Cero':
                    #SIM Izquierda
                    draw_rectangle(ax, inicio= -punto_inicial, ancho= -ancho, direction= direccion, desp= desp, alto= z)
                    #Centro
                    draw_rectangle(ax, inicio= punto_inicial, ancho= ancho, direction= direccion, desp= desp, alto= z)
                    #SIM Derecha
                    draw_rectangle(ax, inicio= -punto_inicial+2*xmax, ancho= -ancho, direction= direccion, desp= desp, alto= z)

                elif pseudo_frontera[0]=='no' or pseudo_frontera[0]=='SIM':
                    #SIM Izquierda
                    draw_rectangle(ax, inicio= -punto_inicial, ancho= -ancho, direction= direccion, desp= desp, alto= z,fill=False)
                    #Centro
                    draw_rectangle(ax, inicio= punto_inicial, ancho= ancho, direction= direccion, desp= desp, alto= z,fill=False)
                    #SIM Derecha
                    draw_rectangle(ax, inicio= -punto_inicial+2*xmax, ancho= -ancho, direction= direccion, desp= desp, alto= z,fill=False)

def graficar_problema_plano_3D(regiones,z=2):
    fronteras = pd.read_csv('csv/' + conf.data['env']['path'] + '/fronteras.csv')
    xmax, ymax = max(regiones['xf']), max(regiones['yf'])

    fig = plt.figure()
    fig.suptitle('Grafica tridimensional del problema plano con 2 simetrias')
    ax = fig.add_subplot(111, projection='3d')

    filename = 'graficas/' + conf.data['env']['path'] + "/Problema Plano 3D.png"

    for i,region in enumerate(regiones.index):
        xi, xf =  regiones.loc[region,'xi'], regiones.loc[region,'xf']
        yi, yf = regiones.loc[region,'yi'], regiones.loc[region,'yf']

        #Izquierda
        draw_region3d(ax,-xi,-xf,yi,yf,fronteras.loc[i],i+1,regiones.loc[region, 'eps'],z,xmax)
        #Central
        draw_region3d(ax,xi,xf,yi,yf,fronteras.loc[i],i+1,regiones.loc[region, 'eps'],z,xmax)
        #Derecha
        draw_region3d(ax,-xi+2*xmax,-xf+2*xmax,yi,yf,fronteras.loc[i],i+1,regiones.loc[region, 'eps'],z,xmax)

    ax.set_xlim(-xmax, 2*xmax)
    ax.set_ylim(0, ymax)
    ax.set_zlim(0, z+2)

    #ax.view_init(60,-60)
    ax.view_init(80,-70)
    fig.set_size_inches(14,8)
    canvas = FigureCanvas(fig)
    canvas.print_figure(filename)

def draw_region(ax, xi, xf, yi, yf, fronteras,n_region,material, xmax,sim='der'):
    x_texto = xi if xi<xf else xf
    desp_t = abs(xf-xi)*.2 if abs(xf-xi)==1 else abs(xf-xi)*.4
    ax.annotate(f'Reg{n_region}\n{material}',(x_texto+desp_t,yi + (yf-yi)*.3))

    for lugar, valor in fronteras.items():

        if lugar=='arriba':
            angulo = 0
            x_t = xi+(xf-xi)*.2 if sim ==None else xi+(xf-xi)*.8
            y_t = yf
        elif lugar=='abajo':
            angulo = 0
            x_t = xi+(xf-xi)*.2 if sim ==None else xi+(xf-xi)*.8
            y_t = yi
        elif lugar=='derecha':
            angulo = 90
            x_t = xf
            y_t = yi+ (yf-yi)*.1
        elif lugar=='izquierda':
            angulo = 90
            x_t = xi
            y_t = yi+ (yf-yi)*.1

        if   valor == 'Uno':  texto = f'V{n_region}=1'
        elif valor == 'Cero': texto = f'V{n_region}=0'
        elif valor == 'SIM':  texto = 'SIM'

        if valor in ['Uno','Cero','SIM']: ax.annotate(texto,(x_t,y_t),rotation=angulo)

        if valor == 'Uno' or valor == 'Cero':
            if lugar == 'arriba':    ax.plot([xi,xf],[yf,yf],'k',lw=3)
            if lugar == 'abajo':     ax.plot([xi,xf],[yi,yi],'k',lw=3)
            if lugar == 'derecha':   ax.plot([xf,xf],[yi,yf],'k',lw=3)
            if lugar == 'izquierda': ax.plot([xi,xi],[yi,yf],'k',lw=3)

        elif valor == 'SIM' or valor == 'no':
            if lugar == 'arriba':    ax.plot([xi,xf],[yf,yf],'r',lw=2)
            if lugar == 'abajo':     ax.plot([xi,xf],[yi,yi],'r',lw=2)
            if lugar == 'derecha':   ax.plot([xf,xf],[yi,yf],'r',lw=2)
            if lugar == 'izquierda': ax.plot([xi,xi],[yi,yf],'r',lw=2)

        else:
            #fronteras Compuestas
            front_list = [x.split('-') for x in valor.split('/')]
            for pseudo_frontera in front_list:
                pi = int(pseudo_frontera[1])
                pf =   int(pseudo_frontera[2])

                if pseudo_frontera[0] == 'Uno' or pseudo_frontera[0] == 'Cero':
                    color ='k'
                    ancho = 3
                elif pseudo_frontera[0] == 'SIM' or pseudo_frontera[0] == 'no':
                    color ='r'
                    ancho = 2

                if lugar == 'arriba':
                    if sim == 'izq': ax.plot([-pi,-pf],[yf,yf],color,lw=ancho)
                    ax.plot([pi,pf],[yf,yf],color,lw=ancho)
                    if sim == 'der':ax.plot([-pi+2*xmax,-pf+2*xmax],[yf,yf],color,lw=ancho)
                if lugar == 'abajo':
                    if sim == 'izq': ax.plot([-pi,-pf],[yi,yi],color,lw=ancho)
                    ax.plot([pi,pf],[yi,yi],color,lw=ancho)
                    if sim == 'der': ax.plot([-pi+2*xmax,-pf+2*xmax],[yi,yi],color,lw=ancho)


def graficar_problema_plano_2D(regiones):
    fronteras = pd.read_csv('csv/' + conf.data['env']['path'] + '/fronteras.csv')
    xmax, ymax = max(regiones['xf']), max(regiones['yf'])

    fig, ax= plt.subplots()
    fig.suptitle('Grafica bidimensional del problema plano con 2 simetrias')
    filename = 'graficas/' + conf.data['env']['path'] + "/Problema Plano 2D.png"


    for i,region in enumerate(regiones.index):
        xi, xf =  regiones.loc[region,'xi'], regiones.loc[region,'xf']
        yi, yf = regiones.loc[region,'yi'], regiones.loc[region,'yf']

        #Izquierda
        draw_region(ax,-xi,-xf,yi,yf,fronteras.loc[i],i+1,regiones.loc[region, 'eps'],xmax,sim='izq')
        #Central
        draw_region(ax,xi,xf,yi,yf,fronteras.loc[i],i+1,regiones.loc[region, 'eps'],xmax,sim=None)
        #Derecha
        draw_region(ax,-xi+2*xmax,-xf+2*xmax,yi,yf,fronteras.loc[i],i+1,regiones.loc[region, 'eps'],xmax,sim='der')

    ax.set_xticks(range(xmax+1))
    ax.set_yticks(range(ymax+1))
    ax.grid()
    fig.set_size_inches(14,8)
    canvas = FigureCanvas(fig)
    canvas.print_figure(filename)
