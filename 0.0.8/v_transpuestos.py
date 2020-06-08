import pandas as pd
import numpy as np

# Solo para efectos de ejemplo del main de este archivo
import region as reg
import eigen as eig

__doc__ = """
...
"""

def cargar_recursos_vectores_transpuestos():
    """
    Se carga la informacion para poder calcular los vectores transpuestos
    """
    # Se crea el df
    recursos_v_transpuestos = pd.read_csv('csv/vectores_transpuestos.csv')
    # Se cambia el nombre de los index al nombre de los vectores transpuestos
    # Se crea un MultiIndex para evitar agregar una columna a los vectores transpuestos
    # queda definida por tuples = [('Reg.1','H'), ('Reg.2', 'F'),('Reg.3', 'G'),('Reg.4', 'H'),...,('Reg.10', 'A')]
    tuples = [('Reg.' + str(i), vt) for i, vt in zip(recursos_v_transpuestos.pop('num_de_region'), recursos_v_transpuestos.pop('chr_vector_t'))]
    recursos_v_transpuestos.index = pd.MultiIndex.from_tuples(tuples)
    return recursos_v_transpuestos


def calcular_vector_transpuesto(x,index_reg_actual,vector_eigen,recursos_v_transpuestos):
    vector_transpuesto = vector_eigen * (x - float(recursos_v_transpuestos.loc[index_reg_actual, 'x_desp']))
    if (recursos_v_transpuestos.loc[index_reg_actual, 'tipo_f']=='sin').bool():
        vector_transpuesto = np.sin(vector_transpuesto)
    else:
        vector_transpuesto = np.cos(vector_transpuesto)
    return vector_transpuesto


def calcular_vectores_transpuestos(vectores_valores_eigen, mesh_regiones, recursos_v_transpuestos, n_dimension=100):
    """
    Se calcula los vectores transpuestos
    """
    # Se crea el df de de vectores transpuestos
    vectores_transpuestos = pd.DataFrame(np.zeros(((len(recursos_v_transpuestos.index)),\
                                         n_dimension)), index=recursos_v_transpuestos.index)
    # Para evitar confusiones los indices de las columnas van desde 1 en adelante
    vectores_transpuestos.columns += 1
    for vector_trans in vectores_transpuestos.index:
        # Reg.1, Reg.2, ... Reg.n - Iteradores de las regiones
        index_reg_actual = vector_trans[0]
        # H, F, I, J .... - Iteradores de los vectores transpuestos
        chr_vector_t = vector_trans[1]
        # Se obtiene el vector plano del meshgrid de x por cada region
        x = mesh_regiones.loc[index_reg_actual,'x']
        # Se calcula primero el argumento del vector transpuesto
        vectores_transpuestos.loc[vector_trans] = vectores_valores_eigen.loc[recursos_v_transpuestos.loc[vector_trans, 'chr_eigen']]\
                                                  * (x - recursos_v_transpuestos.loc[vector_trans, 'x_desp'])
    # Se busca el tipo de funcion( sin o cos) y se le aplica al argumento calculado
    vectores_transpuestos.loc[recursos_v_transpuestos['tipo_f']=='sin',:] = np.sin(vectores_transpuestos.loc[recursos_v_transpuestos['tipo_f']=='sin',:].to_numpy())
    vectores_transpuestos.loc[recursos_v_transpuestos['tipo_f']=='cos',:] = np.cos(vectores_transpuestos.loc[recursos_v_transpuestos['tipo_f']=='cos',:].to_numpy())
    return vectores_transpuestos


def main():
    regiones = reg.cargar_regiones(n_dimension=100)  # 20000000
    # Se crea variable local n_dimension a partir de n_dimension  obtenida del df regiones
    n_dimension = regiones['n_dimension'][0]
    mesh_regiones = reg.cargar_mesh_regiones(regiones, n_dimension)
    valores_eigen = eig.cargar_valores_eigen(regiones)
    vectores_valores_eigen = eig.calcular_vectores_valores_eigen(valores_eigen, n_dimension)
    recursos_v_transpuestos = cargar_recursos_vectores_transpuestos()
    vectores_transpuestos = calcular_vectores_transpuestos(vectores_valores_eigen, mesh_regiones, recursos_v_transpuestos, n_dimension)
    print(f"\t\t\t Vectores de valores eigen")
    print(vectores_valores_eigen)
    print(f"\n\t\t\t Vectores transpuestos")
    print(vectores_transpuestos)
    print(f"\n\t\t\t Vector transpuesto H Reg.1")
    print(vectores_transpuestos.loc['Reg.1', 'H'])
    print(f"\n\t\tVector transpuesto H en numpy de la Reg.1\n")
    print(vectores_transpuestos.loc['Reg.1'])
    print(f"\n\tRecursos para calcular vectores transpuestos")
    print(recursos_v_transpuestos)

if __name__ == '__main__':
    main()
