from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import eigen as eig
import region as reg
import hiperbolica as hyp
import matrices_acomplamiento as m_acop
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

def prueba_valor_eigen(valores_eigen, vectores_valores_eigen, vectores_valores_eigen_err, n_dimension=100, error=1):
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
        # Encuentre el maximo de valor eigen entre el comparado y el original (para efectos de graficacion)
        plt.figure()
        maximo = np.max((vectores_valores_eigen.loc[chr_eigen].max(), vectores_valores_eigen_err.loc[chr_eigen].max()))
        plt.title(f"Control {error_t} nr={n_dimension} del Valor Eigen {chr_eigen+'='+valores_eigen.loc[chr_eigen]['calcular_str']}")
        plt.text(n_dimension / 8, 0.95 * maximo, """Prueba correcta si se imprime una sola curva.
                 Error si imprime dos curvas""")
        # Grafique vector transpuesto de Valores eigen de la funcion (sin error) Color rojo
        plt.plot(N, vectores_valores_eigen.loc[chr_eigen], 'r', label='sin error')
        # Grafique vector transpuesto de Valores eigen de la ecuacion (con error) Color azul
        plt.plot(N, vectores_valores_eigen_err.loc[chr_eigen], 'b', label='con error')
        plt.legend(loc='lower right')
        plt.savefig(f"graficas/vectores eigen/control {error_t}" + " " + chr_eigen + ".svg", bbox_inches='tight')
        plt.close()
        # Salida por consola del proceso que se esta realizando
        print(f"* {chr_eigen+'='+valores_eigen.loc[chr_eigen]['calcular_str']}")


def prueba_matrices_diagonal_valores_eigen(valores_eigen, vectores_valores_eigen, vectores_valores_eigen_err_matriz, n_dimension=100, error=1):
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
        plt.figure()
        # Encuentre el maximo de valor eigen entre el comparado y el original (para efectos de graficacion del texto)
        maximo = np.max((vectores_valores_eigen.loc[chr_eigen].max(), vectores_valores_eigen_err_matriz.loc[chr_eigen].max()))
        plt.title(f"{error_t.capitalize()} - nr={n_dimension} - Matriz diagonal del valor Eigen: {chr_eigen+'='+valores_eigen.loc[chr_eigen]['calcular_str']}")
        plt.text(n_dimension / 8, 0.95 * maximo, """Prueba correcta si se imprime una sola curva.
                 Error si imprime dos curvas""")
        plt.plot(N, vectores_valores_eigen.loc[chr_eigen], 'r', label='sin error')
        plt.plot(N, vectores_valores_eigen_err_matriz.loc[chr_eigen], 'b', label='con error')
        plt.legend(loc='lower right')
        plt.savefig(f"graficas/matrices diagonales de valores eigen/control {error_t}" + " " + chr_eigen + ".svg", bbox_inches='tight')
        plt.close()
        print(f"* Matriz diagonal {chr_eigen+'='+valores_eigen.loc[chr_eigen]['calcular_str']}")


def prueba_matrices_diagonal_funciones_hiperbolicas(funciones_hiperbolicas, vectores_funciones_hiperbolicas, vectores_funciones_hiperbolicas_err, n_dimension=100, error=1):
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
        plt.figure()
        maximo = np.max((vectores_funciones_hiperbolicas.loc[nro_diagonal].max(), vectores_funciones_hiperbolicas_err.loc[nro_diagonal].max()))
        minimo = np.min((vectores_funciones_hiperbolicas.loc[nro_diagonal].min(), vectores_funciones_hiperbolicas_err.loc[nro_diagonal].min()))
        plt.title(f"{error_t.capitalize()} - nr={n_dimension} - Control de la matriz diagonal: {nro_diagonal+'='+funciones_hiperbolicas.loc[nro_diagonal]['calcular_str']}")
        plt.text( 0.1 * n_dimension, minimo + ((maximo - minimo) / 2), """Prueba correcta si se imprime una sola curva.
                 Error si imprime dos curvas""")
        # plt.axvline(0.1 * n_dimension, color='k', linestyle='solid')
        # plt.axhline(.00005 * maximo, color='k', linestyle='solid')
        plt.plot(N, vectores_funciones_hiperbolicas.loc[nro_diagonal], 'r', label='sin error')
        plt.plot(N, vectores_funciones_hiperbolicas_err.loc[nro_diagonal], 'b', label='con error')
        plt.legend(loc='lower right')
        plt.savefig(f"graficas/matrices diagonales de funciones hiperbolicas/control {error_t}" + " " + nro_diagonal + ".svg", bbox_inches='tight')
        plt.close()
        print(f"* Matriz diagonal {nro_diagonal+'='+funciones_hiperbolicas.loc[nro_diagonal]['calcular_str']}")


def prueba_matrices_cuadradas_acoplamiento(integrandos_matrices_acoplamiento, matrices_acomplamiento_int, matrices_acomplamiento_sol, n_dimension=100, error=1):
    """
    Funcion encargada de graficar con matplotlib las matrices cuadradas de aco-
    acoplamiento solucion analitica versus solucion por quad de scipy.
    Equivalente a: f05_Diag_Func_Hiper_Prueba()
    Pametros de entrada:

    Salida:
        * Guarda las figuras en la carpeta "../graficas/matrices cuadradas de
          acomplamiento". Esta carpeta debe estar previamente creada
          para que no haya conflictos al momento de guardar las graficas.
    """
    error_t = 'sin error ' if error == 1 else 'con error'
    N = range(1, (n_dimension * n_dimension) + 1)        # Solo para propositos de graficacion
    # Se obtiene un index a partir del df integrandos_matrices_acoplamiento
    matrices_acomplamiento_sol = error * matrices_acomplamiento_sol
    for M in integrandos_matrices_acoplamiento.index:
        plt.figure()
        # Encuentre el maximo de valor de las dos matrices (la matriz sin error y la matriz con error) (para efectos de graficacion)
        # Nota importante: A cada matriz se le hace un stack durante todo el proceso para obtener un vector de todos los valores de la Matriz
        maximo = np.max((matrices_acomplamiento_int.loc[M].stack().loc[:n_dimension,:n_dimension].max(), matrices_acomplamiento_sol.loc[M].stack().loc[:n_dimension,:n_dimension].max()))
        plt.title(f"{error_t.capitalize()} - nr={n_dimension} - {M + '=' + integrandos_matrices_acoplamiento.loc[M, 'calcular_str']}")
        # plt.text( 0.5 * (n_dimension ** 2), maximo, """Prueba correcta si se imprime una sola grafica.
                 # Error si imprime dos graficas""")
        plt.plot(N, matrices_acomplamiento_int.loc[M].stack().loc[:n_dimension,:n_dimension], 'r', label='sol. integrate.quad')
        plt.plot(N, matrices_acomplamiento_sol.loc[M].stack().loc[:n_dimension,:n_dimension], 'b', label='sol. analitica ' + error_t)
        plt.legend(loc='lower right')
        plt.savefig(f"graficas/matrices cuadradas de acomplamiento/control {error_t}" + " " + M + ".svg", bbox_inches='tight')
        plt.close()
        print(f"* Matriz acompladora {M + '=' + integrandos_matrices_acoplamiento.loc[M, 'calcular_str']}")



def prueba_vectores_distorsionadores(integrandos_vectores_distorsionadores, vectores_distorsionadores_int, vectores_distorsionadores_sol, n_dimension=100, error=1):
    """
    Funcion encargada de graficar con matplotlib los vectores distorsionadores
    versus solucion por quad de scipy.
    Equivalente a: f05_Diag_Func_Hiper_Prueba()
    Pametros de entrada:

    Salida:
        * Guarda las figuras en la carpeta "../graficas/matrices cuadradas de
          acomplamiento". Esta carpeta debe estar previamente creada
          para que no haya conflictos al momento de guardar las graficas.
    """
    error_t = 'sin error ' if error == 1 else 'con error'
    N = range(1, n_dimension + 1)        # Solo para propositos de graficacion
    # Se agrega un error a la solucion analitica
    vectores_distorsionadores_sol = error * vectores_distorsionadores_sol
    # Se obtiene un index a partir del df integrandos_vectores_distorsionadores
    for Sm in integrandos_vectores_distorsionadores.index:
        plt.figure()
        # Encuentre el maximo de valor de las dos matrices (la matriz sin error y la matriz con error) (para efectos de graficacion)
        maximo = np.max((vectores_distorsionadores_int.loc[Sm][:n_dimension].max(), vectores_distorsionadores_sol.loc[Sm][:n_dimension].max()))
        plt.xticks(N)
        plt.title(f"{error_t.capitalize()} - nr={n_dimension} - Vector Dist.: {Sm + '=' + integrandos_vectores_distorsionadores.loc[Sm, 'calcular_str']}")
        # plt.text( 0.5 * (n_dimension ** 2), maximo, """Prueba correcta si se imprime una sola grafica.
                 # Error si imprime dos graficas""")
        plt.plot(N, vectores_distorsionadores_int.loc[Sm][:n_dimension], 'r', label='sol. integrate.quad')
        plt.plot(N, vectores_distorsionadores_sol.loc[Sm][:n_dimension], 'b', label='sol. analitica '+error_t)
        plt.legend(loc='upper right')
        plt.savefig(f"graficas/vectores distorsionadores/control {error_t}" + " " + Sm + ".svg", bbox_inches='tight')
        plt.close()
        print(f"* Vector distorsionador {Sm + '=' + integrandos_vectores_distorsionadores.loc[Sm, 'calcular_str']}")


def prueba_potencial(regiones, recursos_potencial, potenciales, potenciales_err, n_dimension=100, error=1):
    """
    Funcion encargada de graficar con matplotlib los vectores eigen versus los
    vectores eigen calculados con error.Equivalente a : f12_V_dV_Prueba()
    Pametros de entrada:

    Salida:
        * Guarda las figuras en la carpeta ../graficas/potenciales . Esta car
          peta debe estar previamente creada para que no haya conflictos al mo-
          mento de guardar las graficas.
    """
    error_t = 'sin error ' if error == 1 else 'con error'
    N = range(1, n_dimension + 1)        # Solo para propositos de graficacion
    for n_potencial in potenciales.index:
        # Reg.1, Reg.2, ... Reg.n - Iteradores de las regiones
        index_reg_actual = "Reg." + n_potencial.split('V')[1]
        # Encuentre el maximo de valor eigen entre el comparado y el original (para efectos de graficacion)
        plt.figure()
        maximo = np.max((potenciales.loc[n_potencial].max(), potenciales_err.loc[n_potencial].max()))
        minimo = np.max((potenciales.loc[n_potencial].min(), potenciales_err.loc[n_potencial].min()))
        plt.title(f"Con nr={n_dimension}- Prueba del potencial {error_t} de la {index_reg_actual}-{regiones.loc[index_reg_actual, 'eps']}.")
        # plt.text(n_dimension / 8, 0.95 * maximo, """Prueba correcta si se imprime una sola curva.
        #          Error si imprime dos curvas""")
        # Grafique potenciales (con error) Color rojo
        plt.plot(N, potenciales_err.loc[n_potencial], 'r', label='con error')
        # Grafique potenciales (sin error) Color negro
        plt.plot(N, potenciales.loc[n_potencial], 'k', label='sin error')
        plt.legend(loc='lower right')
        plt.savefig(f"graficas/potenciales/control {error_t}" + " " + n_potencial + ".svg", bbox_inches='tight')
        plt.close()
        # Salida por consola del proceso que se esta realizando
        print(f"* {n_potencial}={recursos_potencial.loc[n_potencial,'calcular_str']}")

def prueba_flujo(regiones, recursos_flujo, flujos, flujos_err, n_dimension=100, error=1):
    """
    Funcion encargada de graficar con matplotlib los vectores eigen versus los
    vectores eigen calculados con error.Equivalente a : f12_V_dV_Prueba()
    Pametros de entrada:

    Salida:
        * Guarda las figuras en la carpeta ../graficas/flujos . Esta car
          peta debe estar previamente creada para que no haya conflictos al mo-
          mento de guardar las graficas.
    """
    error_t = 'sin error ' if error == 1 else 'con error'
    N = range(1, n_dimension + 1)        # Solo para propositos de graficacion
    for n_flujo in flujos.index:
        # Reg.1, Reg.2, ... Reg.n - Iteradores de las regiones
        index_reg_actual = "Reg." + n_flujo.split('V')[1]
        # Encuentre el maximo de valor eigen entre el comparado y el original (para efectos de graficacion)
        plt.figure()
        maximo = np.max((flujos.loc[n_flujo].max(), flujos_err.loc[n_flujo].max()))
        minimo = np.max((flujos.loc[n_flujo].min(), flujos_err.loc[n_flujo].min()))
        plt.title(f"Con nr={n_dimension}- Prueba del flujo {error_t} de la {index_reg_actual}-{regiones.loc[index_reg_actual, 'eps']}.")
        # plt.text(n_dimension / 8, 0.95 * maximo, """Prueba correcta si se imprime una sola curva.
        #          Error si imprime dos curvas""")
        # Grafique flujos (con error) Color rojo
        plt.plot(N, flujos_err.loc[n_flujo], 'r', label='con error')
        # Grafique flujos (sin error) Color negro
        plt.plot(N, flujos.loc[n_flujo], 'k', label='sin error')
        plt.legend(loc='lower right')
        plt.savefig(f"graficas/flujos/control {error_t}" + " " + n_flujo + ".svg", bbox_inches='tight')
        plt.close()
        # Salida por consola del proceso que se esta realizando
        print(f"* {n_flujo}={recursos_flujo.loc[n_flujo,'calcular_str']}")


def main():
    # Se cargan las regiones respectivas
    regiones = reg.cargar_regiones()  # region.cargarRegiones(2000000)
    # Carga de dimension a partir de las regiones
    n_dimension = regiones['n_dimension'][0]
    ejes_relativos = reg.cargar_ejes_relativos(regiones)
    mesh_regiones = reg.cargar_mesh_regiones(regiones, n_dimension, delta=10)
    # Carga de los valores eigen significativos de las regiones
    valores_eigen = eig.cargar_valores_eigen(regiones)
    # Calculo de los vectores de los valores eigen
    vectores_valores_eigen = eig.calcular_vectores_valores_eigen(valores_eigen, n_dimension)
    matrices_diagonales_valores_eig = eig.calcular_matrices_diagonal_valores_eigen(vectores_valores_eigen, n_dimension)
    # Calculo de los vectores de los valores eigen de comparacion (con error)
    vectores_valores_eigen_err = eig.calcular_vectores_valores_eigen_error(valores_eigen, n_dimension, 2)
    # Calculo de los vectores de valores eigen de comparacion (con error)
    # este tipo de vector se utilizara para graficar la prueba de matrices de valores eigen con error
    vectores_valores_eigen_err_matriz = 2 * vectores_valores_eigen.abs()
    # Se cargan las funciones hiperbolicas
    funciones_hiperbolicas = hyp.cargar_funciones_hiperbolicas()
    # Se crean dos vectores uno sin error y el otro con error
    vectores_funciones_hiperbolicas = hyp.calcular_vectores_funciones_hiperbolicas(funciones_hiperbolicas, vectores_valores_eigen, n_dimension)
    # Se calculan las matrices diagonales hiperbolicas
    matrices_diagonales_hiperbolicas = hyp.calcular_matrices_diagonales_hiperbolicas(vectores_funciones_hiperbolicas, n_dimension)
    vectores_funciones_hiperbolicas_err = hyp.calcular_vectores_funciones_hiperbolicas(funciones_hiperbolicas, vectores_valores_eigen, n_dimension, error=2)
    # Se carga la informacion necesaria para poder calcular las matrices de acoplamiento
    integrandos_matrices_acoplamiento = m_acop.cargar_integrandos_matrices_acoplamiento()
    # Se calcula las matrices con la funcion quad de scipy solo se hace una muestra para 5 valores eigen de cada vector eigen
    matrices_acomplamiento_int = m_acop.calcular_matrices_acomplamiento_integral(integrandos_matrices_acoplamiento, vectores_valores_eigen.loc[:,0:5], n_dimension=5)
    # Se calcula las matrices con la solucion analitica solo se hace una muestra para 5 valores eigen de cada vector eigen
    matrices_acomplamiento_sol = m_acop.calcular_matrices_acomplamiento_solucion(integrandos_matrices_acoplamiento, vectores_valores_eigen, n_dimension)
    matrices_acomplamiento_trans = m_acop.calcular_matrices_acomplamiento_transpuestas(matrices_acomplamiento_sol)
    # Se carga la informacion necesaria para poder calcular los vectores distorsionadores
    integrandos_vectores_distorsionadores = v_dist.cargar_integrandos_vectores_distorsionadores()
    # Se calcula los vectores distorsionadores con quad de scipy se hace una muestra para solo 10 valores eigen de cada vector eigen
    vectores_distorsionadores_int = v_dist.calcular_vectores_distorsionadores_integral(integrandos_vectores_distorsionadores, vectores_valores_eigen, n_dimension)
    vectores_distorsionadores_sol = v_dist.calcular_vectores_distorsionadores_solucion(integrandos_vectores_distorsionadores, vectores_valores_eigen, n_dimension)
    constantes_c = m_gauss.calcular_matriz_gauss(vectores_valores_eigen, matrices_diagonales_valores_eig, matrices_diagonales_hiperbolicas, matrices_acomplamiento_sol,
                                                 vectores_distorsionadores_sol, matrices_acomplamiento_trans, n_dimension)
    # se cargan los recursos para calcular los vectores transpuestos
    recursos_v_transpuestos = v_trans.cargar_recursos_vectores_transpuestos()
    # se calculan los vectores transpuestos
    # vectores_transpuestos = v_trans.calcular_vectores_transpuestos(vectores_valores_eigen, mesh_regiones, recursos_v_transpuestos, n_dimension)
    recursos_potencial = pot.cargar_recursos_potencial(regiones, ejes_relativos)
    # se cargan los recursos para calcular los potenciales
    potenciales = pot.calcular_potenciales(regiones, ejes_relativos, mesh_regiones, vectores_valores_eigen,\
                                       recursos_v_transpuestos, recursos_potencial, constantes_c, n_dimension)
    potenciales_err = 1.05 * potenciales
    recursos_flujo = flj.cargar_recursos_flujo(regiones, ejes_relativos)
    flujos = flj.calcular_flujos(regiones, ejes_relativos, mesh_regiones, vectores_valores_eigen,\
                                       recursos_v_transpuestos, recursos_flujo, constantes_c, n_dimension)
    flujos_err = 1.05 * flujos
    print("\n\t\t\tValores eigen\n")
    print(valores_eigen)
    print("\n\t\tVectores de Valores eigen\n")
    print(vectores_valores_eigen)
    print("\n\t\tVectores de Valores eigen con error\n")
    print(vectores_valores_eigen_err)
    print("\n\t\tVectores de funciones hiperbolicas\n")
    print(vectores_funciones_hiperbolicas)
    print("\n\t\tVectores de funciones hiperbolicas con error\n")
    print(vectores_funciones_hiperbolicas_err)
    print("\n\t\tMatices de acoplamiento con quad\n")
    print(matrices_acomplamiento_int)
    print("\n\t\tMatices de acoplamiento con sol.analitica\n")
    print(matrices_acomplamiento_sol)
    print("\n\t\t\tPotenciales calculados\n")
    print(potenciales)
    print("\n\t\tPrueba de valores eigen sin error\n")
    prueba_valor_eigen(valores_eigen, vectores_valores_eigen, vectores_valores_eigen, n_dimension)
    print("\n\t\tPrueba de valores eigen con error\n")
    prueba_valor_eigen(valores_eigen, vectores_valores_eigen, vectores_valores_eigen_err, n_dimension, 2)
    print("\n\t\tPrueba de Matrices de valores eigen sin error\n")
    prueba_matrices_diagonal_valores_eigen(valores_eigen, vectores_valores_eigen, vectores_valores_eigen, n_dimension)
    print("\n\t\tPrueba de Matrices de valores eigen con error\n")
    prueba_matrices_diagonal_valores_eigen(valores_eigen, vectores_valores_eigen, vectores_valores_eigen_err_matriz, n_dimension, error=2)
    print("\n\t\tPrueba de Matrices de funciones hiperbolicas sin error\n")
    prueba_matrices_diagonal_funciones_hiperbolicas(funciones_hiperbolicas, vectores_funciones_hiperbolicas, vectores_funciones_hiperbolicas, n_dimension)
    print("\n\t\tPrueba de Matrices de funciones hiperbolicas con error\n")
    prueba_matrices_diagonal_funciones_hiperbolicas(funciones_hiperbolicas, vectores_funciones_hiperbolicas, vectores_funciones_hiperbolicas_err, n_dimension, error=2)
    print("\n\tPrueba de Matrices de cuadradas de acoplamiento sin error\n")
    prueba_matrices_cuadradas_acoplamiento(integrandos_matrices_acoplamiento, matrices_acomplamiento_int, matrices_acomplamiento_sol, n_dimension=5)
    print("\n\tPrueba de Matrices de cuadradas de acoplamiento con error\n")
    prueba_matrices_cuadradas_acoplamiento(integrandos_matrices_acoplamiento, matrices_acomplamiento_int, matrices_acomplamiento_sol, n_dimension=5, error=2)
    print("\n\tPrueba de Vectores distorsionadores sin error\n")
    prueba_vectores_distorsionadores(integrandos_vectores_distorsionadores, vectores_distorsionadores_int, vectores_distorsionadores_sol, n_dimension=10)
    print("\n\tPrueba de Vectores distorsionadores con error\n")
    prueba_vectores_distorsionadores(integrandos_vectores_distorsionadores, vectores_distorsionadores_int, vectores_distorsionadores_sol, n_dimension=10, error=2)
    print("\n\t\tPrueba de Potenciales sin error\n")
    prueba_potencial(regiones, recursos_potencial, potenciales, potenciales, n_dimension)
    print("\n\t\tPrueba de Potenciales con error\n")
    prueba_potencial(regiones, recursos_potencial, potenciales, potenciales_err, n_dimension,error=1.05)
    print("\n\t\tPrueba de Flujos sin error\n")
    prueba_flujo(regiones, recursos_flujo, flujos, flujos, n_dimension)
    print("\n\t\tPrueba de flujos con error\n")
    prueba_flujo(regiones, recursos_flujo, flujos, flujos_err, n_dimension,error=1.05)

if __name__ == '__main__':
    main()
