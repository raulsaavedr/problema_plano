import pandas as pd
import numpy as np

from config import conf

__doc__ = """
...
"""

def cargar_recursos_vectores_transpuestos():
    """
    Se carga la informacion para poder calcular los vectores transpuestos
    """
    # Se crea el df
    filename = 'csv/' + conf.data['env']['path'] + '/vectores_transpuestos.csv'
    recursos_v_transpuestos = pd.read_csv(filename)
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
