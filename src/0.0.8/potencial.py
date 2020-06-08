import numpy as np
import pandas as pd

import eigen as eig
import region as reg
import hiperbolica as hyp
import matrices_acomplamiento as m_acop
import distorsionador as v_dist
import matriz_gauss as m_gauss
import v_transpuestos as v_trans


__doc__ = """
Este modulo se determina el potencial y la continuidad de potencial.
Es equivalente a la funcion: * f11_V(region,modo,x,y)
"""

def cargar_recursos_potencial():
    """
    Funcion encargada de crear un dataframe que funciona como esquema o patron
    para luego poder hacer el calculo del potencial
    """
    recursos_potencial = pd.read_csv('csv/artificio_potencial.csv')
    recursos_potencial.index = ['V' + str(i) for i in range(1, len(recursos_potencial) + 1)]
    return recursos_potencial



def calcular_potenciales(regiones, ejes_relativos, mesh_regiones, vectores_valores_eigen,\
                         recursos_v_transpuestos, recursos_potencial, constantes_c, n_dimension=100):
    """
    Funcion encargada de construir los potenciales.
    Parametros de entrada:
        * regiones: Es un dataframe donde se encuentran almacenados todos los
          valores importantes de las regiones tales como: limites de 'x' y 'y',
          tipo de material(Eps o E0), tipo de funcion ('sin' o 'cos'), etc.
    Salida:
        * potenciales: Es un dataframe que contiene el potencial en todas las
          regiones.
    """
    # Se crea el DataFrame de dimension = valores_eigen*n_dimension y se llena con ceros
    potenciales = pd.DataFrame(np.zeros(((len(recursos_potencial.index)), n_dimension)), index=recursos_potencial.index)
    print(f"\n{'-'*35}DEBUG{'-'*35}")
    for n_potencial in potenciales.index:
        # Reg.1, Reg.2, ... Reg.n - Iteradores de las regiones
        index_reg_actual = "Reg." + n_potencial.split('V')[1]
        # Se obtiene el vector plano del meshgrid de (y) por cada region
        x_flat = mesh_regiones.loc[index_reg_actual,'x'].to_numpy()
        y_flat = mesh_regiones.loc[index_reg_actual,'y'].to_numpy()
        # Vector eigen pertinente a la region actual
        vector_eig = vectores_valores_eigen.loc[regiones.loc[index_reg_actual, 'chr_eigen']].to_numpy()
        C1 = np.zeros(n_dimension)[:,np.newaxis]
        if recursos_potencial.loc[n_potencial, 'factor_C_1'] != '0':
            C1 = constantes_c[recursos_potencial.loc[n_potencial, 'factor_C_1']].to_numpy()[:,np.newaxis]
        C2 = np.zeros(n_dimension)[:,np.newaxis]
        if recursos_potencial.loc[n_potencial, 'factor_C_2'] != '0':
            C2 = constantes_c[recursos_potencial.loc[n_potencial, 'factor_C_2']].to_numpy()[:,np.newaxis]
        for i in range(0, n_dimension):
            # Se obtiene el valor i-esimo del vector aplanado del meshgrid en (x) y en y
            x = x_flat[i]
            y = y_flat[i]
            # factor polinomio (x-ca)/(cb-ca)
            factor_poly = 0
            if recursos_potencial.loc[n_potencial, 'term. (x-ca/cb-ca)'] == 'si':
                factor_poly = (x - recursos_potencial.loc[n_potencial, 'ca'])\
                              / (recursos_potencial.loc[n_potencial, 'cb'] - recursos_potencial.loc[n_potencial, 'ca'])
            vector_transpuesto = v_trans.calcular_vector_transpuesto(x,index_reg_actual,vector_eig,recursos_v_transpuestos)
            factor_normalizador_1 = np.diag(np.zeros(n_dimension))
            if recursos_potencial.loc[n_potencial, 'tipo_f1'] != '0':
                # Se calcula el numerador del factor normalizador , puede ser:
                # sinh(y-yrel) o cosh(y-yrel)
                func_num_1 = 'np.' + recursos_potencial.loc[n_potencial, 'tipo_f1'] + '(vector_eig *('\
                             + ejes_relativos.loc[index_reg_actual, 'calc_eje_rel'] + '))'
                # Se calcula el denominador del factor normalizador , puede ser:
                # dependiendo del flujo en esa region sinh(ctte) o cosh(ctte)
                func_denom_1 = 'np.' + recursos_potencial.loc[n_potencial, 'tipo_f1'] + '(vector_eig *('\
                               + ejes_relativos.loc[index_reg_actual, 'ctte_eje_rel_norm'].astype(str) + '))'
                factor_normalizador_1 = np.diag(eval(func_num_1) / eval(func_denom_1))
            factor_normalizador_2 = np.diag(np.zeros(n_dimension))
            if recursos_potencial.loc[n_potencial, 'tipo_f2'] != '0':
                # Se calcula el numerador del factor normalizador , puede ser:
                # sinh(y-yrel) o cosh(y-yrel)
                func_num_2 = 'np.' + recursos_potencial.loc[n_potencial, 'tipo_f2'] + '(vector_eig *('\
                             + ejes_relativos.loc[index_reg_actual, 'calc_eje_rel'] + '))'
                # Se calcula el denominador del factor normalizador , puede ser:
                # dependiendo del flujo en esa region sinh(ctte) o cosh(ctte)
                func_denom_2 = 'np.' + recursos_potencial.loc[n_potencial, 'tipo_f2'] + '(vector_eig *('\
                               + ejes_relativos.loc[index_reg_actual, 'ctte_eje_rel_norm'].astype(str) + '))'
                factor_normalizador_2 = np.diag(eval(func_num_2) / eval(func_denom_2))
            # Se calculan los potenciales
            potenciales.loc[n_potencial, i] = recursos_potencial.loc[n_potencial, 'sum. 1er termino'] - factor_poly\
                                              + vector_transpuesto @ (factor_normalizador_1 @ C1 + factor_normalizador_2 @ C2)
        # print(f"\n{index_reg_actual}")
        # print(f"\nPrimer termino\n{recursos_potencial.loc[n_potencial, 'sum. 1er termino'] - factor_poly}")
        # print(f"\nVector Eigen shape:{vector_eig.shape}\n{vector_eig}")
        # print(f"\nfactor_normalizador_1 shape:{factor_normalizador_1.shape}\n{factor_normalizador_1}")
        # print(f"\nfactor_normalizador_2 shape:{factor_normalizador_2.shape}\n{factor_normalizador_2}")
        # print(f"\nConstante 1 shape:{C1.shape}\n{C1}")
        # print(f"\nConstante 2 shape:{C2.shape}\n{C2}")
        # print(f"\nVector transpuesto shape:{vector_transpuesto.shape}\n{vector_transpuesto}")
    # print(f"\n{'-'*30}END OF DEBUG{'-'*30}")
    # Para evitar confusiones los indices de las columnas van desde 1 en adelante
    potenciales.columns += 1
    return potenciales


def main():
    regiones = reg.cargar_regiones(n_dimension=100)  # 20000000
    # Se crea variable local n_dimension a partir de n_dimension  obtenida del df regiones
    n_dimension = regiones['n_dimension'][0]
    ejes_relativos = reg.cargar_ejes_relativos(regiones)
    mesh_regiones = reg.cargar_mesh_regiones(regiones, n_dimension, delta=10)
    # Carga de los valores eigen significativos de las regiones
    valores_eigen = eig.cargar_valores_eigen(regiones)
    # Calculo de los vectores de los valores eigen
    vectores_valores_eigen = eig.calcular_vectores_valores_eigen(valores_eigen, n_dimension)
    matrices_diagonales_valores_eig = eig.calcular_matrices_diagonal_valores_eigen(vectores_valores_eigen, n_dimension)
    # Se cargan las funciones hiperbolicas
    funciones_hiperbolicas = hyp.cargar_funciones_hiperbolicas()
    # Se crean dos vectores uno sin error y el otro con error
    vectores_funciones_hiperbolicas = hyp.calcular_vectores_funciones_hiperbolicas(funciones_hiperbolicas, vectores_valores_eigen, n_dimension)
    matrices_diagonales_hiperbolicas = hyp.calcular_matrices_diagonales_hiperbolicas(vectores_funciones_hiperbolicas, n_dimension)
    # Se carga la informacion necesaria para poder calcular las matrices de acoplamiento
    integrandos_matrices_acoplamiento = m_acop.cargar_integrandos_matrices_acoplamiento()
    # Se calcula las matrices con la solucion analitica
    matrices_acomplamiento_sol = m_acop.calcular_matrices_acomplamiento_solucion(integrandos_matrices_acoplamiento, vectores_valores_eigen, n_dimension)
    matrices_acomplamiento_trans = m_acop.calcular_matrices_acomplamiento_transpuestas(matrices_acomplamiento_sol)
    # Se carga la informacion necesaria para poder calcular los vectores distorsionadores
    integrandos_vectores_distorsionadores = v_dist.cargar_integrandos_vectores_distorsionadores()
    # Se calcula los vectores distorsionadores con quad de scipy se hace una muestra para solo 10 valores eigen de cada vector eigen
    vectores_distorsionadores_sol = v_dist.calcular_vectores_distorsionadores_solucion(integrandos_vectores_distorsionadores, vectores_valores_eigen, n_dimension)
    # ordenar_constantes()
    constantes_c = m_gauss.calcular_matriz_gauss(vectores_valores_eigen, matrices_diagonales_valores_eig, matrices_diagonales_hiperbolicas, matrices_acomplamiento_sol,
                                                 vectores_distorsionadores_sol, matrices_acomplamiento_trans, n_dimension)
    # se cargan los recursos para calcular los vectores transpuestos
    recursos_v_transpuestos = v_trans.cargar_recursos_vectores_transpuestos()
    # se calculan los vectores transpuestos
    # vectores_transpuestos = v_trans.calcular_vectores_transpuestos(vectores_valores_eigen, mesh_regiones, recursos_v_transpuestos, n_dimension)
    recursos_potencial = cargar_recursos_potencial()
    # se cargan los recursos para calcular los potenciales
    potenciales = calcular_potenciales(regiones, ejes_relativos, mesh_regiones, vectores_valores_eigen,\
                                       recursos_v_transpuestos, recursos_potencial, constantes_c, n_dimension)

    print("\n\t\tConstantes C calculadas\n")
    print(constantes_c)
    print(f"\n\t\t\t Ejes relativos")
    print(ejes_relativos)
    print(f"\n\t\t\t Recursos para calcular potenciales")
    print(recursos_potencial.to_string())
    print(f"\n\t\t\t\t\t Potenciales 1:15")
    print(potenciales.loc[:,1:15].to_string())
    print(f"\n\t\t\t\t\t Potenciales 16:31")
    print(potenciales.loc[:,16:31].to_string())
    print(f"\n\t\t\t\t\t Potenciales 32:47")
    print(potenciales.loc[:,32:47].to_string())
    print(f"\n\t\t\t\t\t Potenciales 47:62")
    print(potenciales.loc[:,48:63].to_string())
    print(f"\n\t\t\t\t\t Potenciales 63:78")
    print(potenciales.loc[:,64:79].to_string())
    print(f"\n\t\t\t\t\t Potenciales 79:100")
    print(potenciales.loc[:,80:100].to_string())


if __name__ == '__main__':
    main()
