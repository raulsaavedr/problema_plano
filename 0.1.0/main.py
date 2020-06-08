import time
init_time = time.time()

from datetime import timedelta
import region as reg
import graficacion as grf
import eigen as eig
import hiperbolica as hyp
import matrices_acomplamiento as m_acop
import distorsionador as v_dist
import matriz_gauss as m_gauss
import v_transpuestos as v_trans
import potencial as pot
import flujo as flj


__doc__ = """
Este programa ha sido creado con python3.7, pandas 0.25.3, numpy 1.16.2,
numexpr 2.7.0. Se utilizan todos los modulos para poder resolver el problema
plano
"""


def main():
    # Se cargan las regiones respectivas
    regiones = reg.cargar_regiones(n_dimension=100)  # region.cargarRegiones(2000000)
    # Carga de dimension a partir de las regiones
    n_dimension = regiones['n_dimension'][0]
    # Se cargan los ejes relativos para poder calcular flujo y potencial
    ejes_relativos = reg.cargar_ejes_relativos(regiones)
    # Se calcula el mesh para cada region del problema
    mesh_regiones = reg.cargar_mesh_regiones(regiones, n_dimension, delta=10)
    # Carga de los valores eigen significativos de las regiones
    valores_eigen = eig.cargar_valores_eigen(regiones)
    # Calculo de los vectores de los valores eigen
    vectores_valores_eigen = eig.calcular_vectores_valores_eigen(valores_eigen, n_dimension)
    # Calculo de los vectores de los valores eigen de comparacion (con error)
    vectores_valores_eigen_err = eig.calcular_vectores_valores_eigen_error(valores_eigen, n_dimension, 2)
    # Calculo de los vectores de valores eigen de comparacion (con error)
    # este tipo de vector se utilizara para graficar la prueba de matrices de valores eigen con error
    vectores_valores_eigen_err_matriz = 2 * vectores_valores_eigen.abs()
    matrices_diagonales_valores_eig = eig.calcular_matrices_diagonal_valores_eigen(vectores_valores_eigen, n_dimension)
    # Se cargan las funciones hiperbolicas
    funciones_hiperbolicas = hyp.cargar_funciones_hiperbolicas()
    # Se crean dos vectores uno sin error y el otro con error
    vectores_funciones_hiperbolicas = hyp.calcular_vectores_funciones_hiperbolicas(funciones_hiperbolicas, vectores_valores_eigen, n_dimension)
    vectores_funciones_hiperbolicas_err = hyp.calcular_vectores_funciones_hiperbolicas(funciones_hiperbolicas, vectores_valores_eigen, n_dimension, error=2)
    matrices_diagonales_hiperbolicas = hyp.calcular_matrices_diagonales_hiperbolicas(vectores_funciones_hiperbolicas, n_dimension)
    # Se carga la informacion necesaria para poder calcular las matrices de acoplamiento
    integrandos_matrices_acoplamiento = m_acop.cargar_integrandos_matrices_acoplamiento()
    # Se calcula las matrices con la funcion quad de scipy solo se hace una muestra para 5 valores eigen de cada vector eigen
    matrices_acomplamiento_int = m_acop.calcular_matrices_acomplamiento_integral(integrandos_matrices_acoplamiento, vectores_valores_eigen, n_dimension)
    # Se calcula las matrices con la solucion analitica solo se hace una muestra para 5 valores eigen de cada vector eigen
    matrices_acomplamiento_sol = m_acop.calcular_matrices_acomplamiento_solucion(integrandos_matrices_acoplamiento, vectores_valores_eigen, n_dimension)
    matrices_acomplamiento_trans = m_acop.calcular_matrices_acomplamiento_transpuestas(matrices_acomplamiento_sol)
    # Se carga la informacion necesaria para poder calcular los vectores distorsionadores
    integrandos_vectores_distorsionadores = v_dist.cargar_integrandos_vectores_distorsionadores()
    # Se calcula los vectores distorsionadores con quad de scipy se hace una muestra para solo 10 valores eigen de cada vector eigen
    vectores_distorsionadores_int = v_dist.calcular_vectores_distorsionadores_integral(integrandos_vectores_distorsionadores, vectores_valores_eigen, n_dimension)
    vectores_distorsionadores_sol = v_dist.calcular_vectores_distorsionadores_solucion(integrandos_vectores_distorsionadores, vectores_valores_eigen, n_dimension)
    # Se hace el calculo de la matriz de gauss y se almacena el resultado(constantes C1,C2,..CN) en constantes_c
    constantes_c = m_gauss.calcular_matriz_gauss(vectores_valores_eigen, matrices_diagonales_valores_eig, matrices_diagonales_hiperbolicas, matrices_acomplamiento_sol,
                                                 vectores_distorsionadores_sol, matrices_acomplamiento_trans, n_dimension)
    # se cargan los recursos para calcular los vectores transpuestos
    recursos_v_transpuestos = v_trans.cargar_recursos_vectores_transpuestos()
    # se cargan los recursos para calcular los potenciales
    recursos_potencial = pot.cargar_recursos_potencial(regiones, ejes_relativos)
    # Se calculan potenciales
    potenciales = pot.calcular_potenciales(regiones, ejes_relativos, mesh_regiones, vectores_valores_eigen,\
                                           recursos_v_transpuestos, recursos_potencial, constantes_c, n_dimension)
    potenciales_err = 1.05 * potenciales
    # se cargan los recursos para calcular los flujos
    recursos_flujo = flj.cargar_recursos_flujo(regiones, ejes_relativos)
    # Se calculan flujos
    flujos = flj.calcular_flujos(regiones, ejes_relativos, mesh_regiones, vectores_valores_eigen,\
                                       recursos_v_transpuestos, recursos_flujo, constantes_c, n_dimension)
    flujos_err = 1.05 * flujos
    print("\n\t\tValores eigen\n")
    print(valores_eigen)
    print("\n\t\tVectores de Valores eigen\n")
    print(vectores_valores_eigen)
    print("\n\t\tFunciones hiperbolicas\n")
    print(funciones_hiperbolicas)
    print("\n\t\tVectores de funciones hiperbolicas\n")
    print(vectores_funciones_hiperbolicas)
    print("\n\t\t\tConstantes C calculadas\n")
    print(constantes_c)
    print(f"\n\t\t\tPotenciales calculados")
    print(potenciales)
    print(f"\n\t\t\tFlujos calculados")
    print(flujos)
    print("\t\tPrueba de vectores de valores eigen sin error\n")
    grf.prueba_valor_eigen(valores_eigen, vectores_valores_eigen, vectores_valores_eigen, n_dimension)
    print("\n\t\tPrueba de valores eigen con error\n")
    grf.prueba_valor_eigen(valores_eigen, vectores_valores_eigen, vectores_valores_eigen_err, n_dimension, 2)
    print("\n\t\tPrueba de Matrices de valores eigen sin error\n")
    grf.prueba_matrices_diagonal_valores_eigen(valores_eigen, vectores_valores_eigen, vectores_valores_eigen, n_dimension)
    print("\n\t\tPrueba de Matrices de valores eigen con error\n")
    grf.prueba_matrices_diagonal_valores_eigen(valores_eigen, vectores_valores_eigen, vectores_valores_eigen_err_matriz, n_dimension, error=2)
    print("\n\t\tPrueba de Matrices de funciones hiperbolicas sin error\n")
    grf.prueba_matrices_diagonal_funciones_hiperbolicas(funciones_hiperbolicas, vectores_funciones_hiperbolicas, vectores_funciones_hiperbolicas, n_dimension)
    print("\n\t\tVectores de funciones hiperbolicas con error\n")
    print(vectores_funciones_hiperbolicas_err)
    print("\n\t\tPrueba de Matrices de funciones hiperbolicas con error\n")
    grf.prueba_matrices_diagonal_funciones_hiperbolicas(funciones_hiperbolicas, vectores_funciones_hiperbolicas, vectores_funciones_hiperbolicas_err, n_dimension, error=2)
    print("\n\t\tMatices de acoplamiento con quad\n")
    print(matrices_acomplamiento_int)
    print("\n\t\tMatices de acoplamiento con sol.analitica\n")
    print(matrices_acomplamiento_sol)
    print("\n\tPrueba de Matrices cuadradas de acoplamiento sin error\n")
    grf.prueba_matrices_cuadradas_acoplamiento(integrandos_matrices_acoplamiento, matrices_acomplamiento_int, matrices_acomplamiento_sol, n_dimension=5)
    print("\n\tPrueba de Matrices cuadradas de acoplamiento con error\n")
    grf.prueba_matrices_cuadradas_acoplamiento(integrandos_matrices_acoplamiento, matrices_acomplamiento_int, matrices_acomplamiento_sol, n_dimension=5, error=2)
    print("\n\t\t\tVectores distorsionadores\n")
    print(vectores_distorsionadores_sol)
    print("\n\tPrueba de Vectores distorsionadores sin error\n")
    grf.prueba_vectores_distorsionadores(integrandos_vectores_distorsionadores, vectores_distorsionadores_int, vectores_distorsionadores_sol, n_dimension=10)
    print("\n\tPrueba de Vectores distorsionadores con error\n")
    grf.prueba_vectores_distorsionadores(integrandos_vectores_distorsionadores, vectores_distorsionadores_int, vectores_distorsionadores_sol, n_dimension=10, error=2)
    print("\n\t\tPrueba de Potenciales sin error\n")
    grf.prueba_potencial(regiones, recursos_potencial, potenciales, potenciales, n_dimension)
    print("\n\t\tPrueba de Potenciales con error\n")
    grf.prueba_potencial(regiones, recursos_potencial, potenciales, potenciales_err, n_dimension,error=1.05)
    print("\n\t\tPrueba de Flujos sin error\n")
    grf.prueba_flujo(regiones, recursos_flujo, flujos, flujos, n_dimension)
    print("\n\t\tPrueba de Flujos con error\n")
    grf.prueba_flujo(regiones, recursos_flujo, flujos, flujos_err, n_dimension,error=1.05)

if __name__ == '__main__':
    # Se ejecuta el calculo del problema plano
    main()
    finish_time = time.time() - init_time
    print(f"\nTiempo de ejecucion {timedelta(seconds=finish_time)} ...")
