import numpy as np
import pandas as pd
import logging

from config import conf
logger = logging.getLogger('potencial')
logger.setLevel(conf.get_logging_level('potencial'))
logger.disabled = conf.data['debugging']['potencial']['disabled']

import eigen as eig
import region as reg
import hiperbolica as hyp
import matrices_acoplamiento as m_acop
import distorsionador as v_dist
import matriz_gauss as m_gauss
import v_transpuestos as v_trans


__doc__ = """
Este modulo se determina el potencial y la continuidad de potencial.
Es equivalente a la funcion: f11_V(region,modo,x,y)
"""

def cargar_recursos_potencial(regiones, ejes_relativos):
    """
    Funcion encargada de crear un dataframe que funciona como esquema o patron
    para luego poder hacer el calculo del potencial
    """
    filename = 'csv/' + conf.data['env']['path'] + '/artificio_potencial.csv'
    recursos_potencial = pd.read_csv(filename)
    recursos_potencial.index = ['V' + str(i) for i in range(1, len(recursos_potencial) + 1)]
    factor_1 = recursos_potencial['sum. 1er termino'].astype(str)
    factor_poly = pd.DataFrame(index=recursos_potencial.index)
    factor_poly.loc[recursos_potencial['term. (x-ca/cb-ca)']=='si','calcular_str'] = "((x-x" + recursos_potencial['ca'].astype(str)\
                                                                                     + ")/(x" + recursos_potencial['cb'].astype(str)\
                                                                                     + "-x" + recursos_potencial['ca'].astype(str) + ")) + "
    factor_poly.loc[recursos_potencial['term. (x-ca/cb-ca)']=='no','calcular_str'] = ""
    factor_normalizador_1 = pd.DataFrame(index=recursos_potencial.index)
    factor_normalizador_1['calcular_str'] = "(" + recursos_potencial['tipo_f1'] + "(" + regiones['chr_eigen'].to_numpy() + "("\
                                            + ejes_relativos['calc_eje_rel_str'].to_numpy() + "))/"\
                                            + recursos_potencial['tipo_f1'] + "(" + regiones['chr_eigen'].to_numpy() + "("\
                                            + ejes_relativos['ctte_eje_rel_norm_str'].to_numpy() + ")))*"
    factor_normalizador_2 = pd.DataFrame(index=recursos_potencial.index)

    factor_normalizador_2.loc[recursos_potencial['factor_C_2']!='0', 'calcular_str'] = " + (" + recursos_potencial['tipo_f2'] + "("\
                                                                                        + ejes_relativos['calc_eje_rel_str'].to_numpy() + ")/"\
                                                                                        + recursos_potencial['tipo_f2'] + "("\
                                                                                        + ejes_relativos['ctte_eje_rel_norm_str'].to_numpy() + "))*"
    factor_normalizador_2.loc[recursos_potencial['factor_C_2']=='0', 'calcular_str'] = ""
    factor_C_2 = pd.DataFrame(index=recursos_potencial.index)
    factor_C_2.loc[recursos_potencial['factor_C_2']!='0', 'calcular_str'] = recursos_potencial['factor_C_2'].astype(str)
    factor_C_2.loc[recursos_potencial['factor_C_2']=='0', 'calcular_str'] = ""
    recursos_potencial['calcular_str'] =  recursos_potencial['sum. 1er termino'].astype(str)\
                                          + " + " + factor_poly['calcular_str']\
                                          + recursos_potencial['chr_vector_t'].astype(str) + u"\u1D40(x)["\
                                          + factor_normalizador_1['calcular_str'] + recursos_potencial['factor_C_1']\
                                          + factor_normalizador_2['calcular_str'] + factor_C_2['calcular_str'] + "]"
    return recursos_potencial



def calcular_potenciales(regiones, ejes_relativos, mesh_regiones, vectores_valores_eigen,\
                         recursos_v_transpuestos, recursos_potencial, constantes_c,\
                         dimension_mesh=100, n_dimension=100):
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
    # Se crea el DataFrame de dimension = valores_eigen*dimension_mesh y se llena con ceros
    potenciales = pd.DataFrame(np.zeros(((len(recursos_potencial.index)), dimension_mesh)), index=recursos_potencial.index)
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
# print("\n\t\t\t Region:%s\n" % index_reg_actual)
        for i in range(0, dimension_mesh):
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
# print("factor_num_1:%s\n" % func_num_1)
# print("func_denom_1:%s\n" % func_denom_1)
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
# print("factor_num_2:%s\n" % func_num_2)
# print("func_denom_2:%s\n" % func_denom_2)
            # Se calculan los potenciales
            potenciales.loc[n_potencial, i] = recursos_potencial.loc[n_potencial, 'sum. 1er termino'] - factor_poly\
                                              + vector_transpuesto @ (factor_normalizador_1 @ C1 + factor_normalizador_2 @ C2)
    # Para evitar confusiones los indices de las columnas van desde 1 en adelante
    potenciales.columns += 1
    return potenciales
