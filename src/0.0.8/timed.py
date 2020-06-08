import timeit

mysetup = """import region as reg
import graficacion as grf
import eigen as eig
import hiperbolica as hyp
import distorsionador as dist"""


my_code = """
def main():
    # Se cargan las regiones respectivas
    regiones = reg.cargar_regiones()  # region.cargarRegiones(2000000)
    # Carga de dimension a partir de las regiones
    n_dimension = regiones['n_dimension'][0]
    # Carga de los valores eigen significativos de las regiones
    valores_eigen = eig.cargar_valores_eigen(regiones)
    # Calculo de los vectores de los valores eigen
    vectores_valores_eigen = eig.calcular_vectores_valores_eigen(valores_eigen, n_dimension)
    # Calculo de los vectores de los valores eigen de comparacion (con error)
    vectores_valores_eigen_err = eig.calcular_vectores_valores_eigen_error(valores_eigen, n_dimension, 2)
    # Calculo de los vectores de valores eigen de comparacion (con error)
    # este tipo de vector se utilizara para graficar la prueba de matrices de valores eigen con error
    vectores_valores_eigen_err_matriz = 2 * vectores_valores_eigen.abs()
    # Se cargan las funciones hiperbolicas
    funciones_hiperbolicas = hyp.cargar_funciones_hiperbolicas()
    # Se crean dos vectores uno sin error y el otro con error
    vectores_funciones_hiperbolicas = hyp.calcular_vectores_funciones_hiperbolicas(funciones_hiperbolicas, vectores_valores_eigen, n_dimension)
    vectores_funciones_hiperbolicas_err = hyp.calcular_vectores_funciones_hiperbolicas(funciones_hiperbolicas, vectores_valores_eigen, n_dimension, error=2)

# Se ejecuta el calculo del problema plano
main()
"""

elapsed_time = timeit.timeit(setup=mysetup, stmt=my_code, number=3) / 3
print(f"Time code:  {elapsed_time}")
