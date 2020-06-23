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


__doc__ = """
Este modulo se determina el flujo y la continuidad de flujo.
Es equivalente a la funcion: * f11_V(region,modo,x,y)
"""

def cargar_recursos_flujo(regiones, ejes_relativos):
    """
    Funcion encargada de crear un dataframe que funciona como esquema o patron
    para luego poder hacer el calculo de los flujos en cada region
    """
    filename = 'csv/' + conf.data['env']['path'] + '/artificio_potencial.csv'
    recursos_flujo = pd.read_csv(filename)
    # Se eliman algunas columnas de potencial que ya no son necesarias
    recursos_flujo.drop(['sum. 1er termino', 'term. (x-ca/cb-ca)', 'ca', 'cb'], inplace=True, axis=1)
    recursos_flujo.rename(columns={'tipo_f1':'tipo_f1_num','tipo_f2':'tipo_f2_num'}, inplace=True)
    # Se agregan nuevas columnas utiles para recursos_flujo
    recursos_flujo.insert(0, 'signo', '')
    recursos_flujo.insert(4, 'tipo_f1_denom', '')
    recursos_flujo.insert(5, 'tipo_f2_denom', '')
    # Para poder utilzar la busqueda condicional de pandas se deja el mismo index
    recursos_flujo.index = regiones.index
    recursos_flujo.loc[regiones['direcc_de_flujo']=='subiendo','signo'] = '+'
    recursos_flujo.loc[regiones['direcc_de_flujo']=='bajando','signo'] = '-'
    recursos_flujo.loc[regiones['direcc_de_flujo']=='ambos sentidos','signo'] = '+'
    # Se construye las funciones del factor normalizador 1
    recursos_flujo.loc[recursos_flujo['tipo_f1_num']=='sinh','tipo_f1_num'] = 'cosh1'
    recursos_flujo.loc[recursos_flujo['tipo_f1_num']=='cosh','tipo_f1_num'] = 'sinh'
    recursos_flujo.loc[recursos_flujo['tipo_f1_num']=='cosh1','tipo_f1_num'] = 'cosh'
    recursos_flujo.loc[recursos_flujo['tipo_f1_num']=='cosh','tipo_f1_denom'] = 'sinh'
    recursos_flujo.loc[recursos_flujo['tipo_f1_num']=='sinh','tipo_f1_denom'] = 'cosh'
    # Se construye las funciones del factor normalizador 2
    recursos_flujo.loc[recursos_flujo['tipo_f2_num']=='0','tipo_f2_denom'] = '1'  # Previniendo futura indeterminacion
    recursos_flujo.loc[recursos_flujo['tipo_f2_num']=='sinh','tipo_f2_num'] = 'cosh1'
    recursos_flujo.loc[recursos_flujo['tipo_f2_num']=='cosh','tipo_f2_num'] = 'sinh'
    recursos_flujo.loc[recursos_flujo['tipo_f2_num']=='cosh1','tipo_f2_num'] = 'cosh'
    recursos_flujo.loc[recursos_flujo['tipo_f2_num']=='cosh','tipo_f2_denom'] = 'sinh'
    recursos_flujo.loc[recursos_flujo['tipo_f2_num']=='sinh','tipo_f2_denom'] = 'cosh'
    # Se agrega el index que corresponde a los flujos
    recursos_flujo.index = [u'\u2202V' + str(i) for i in range(1, len(recursos_flujo) + 1)]
    # Se crean nuevas columnas calcular_str para mostrar en graficacion.py
    factor_signo = pd.DataFrame(index=recursos_flujo.index)
    factor_signo.loc[recursos_flujo['signo']=='+', 'calcular_str'] = ""
    factor_signo.loc[recursos_flujo['signo']=='-', 'calcular_str'] = "-"
    factor_normalizador_1 = pd.DataFrame(index=recursos_flujo.index)
    factor_normalizador_1['calcular_str'] = "(" + recursos_flujo['tipo_f1_num'] + "(" + regiones['chr_eigen'].to_numpy() + "("\
                                            + ejes_relativos['calc_eje_rel_str'].to_numpy() + "))/"\
                                            + recursos_flujo['tipo_f1_denom'] + "(" + regiones['chr_eigen'].to_numpy() + "("\
                                            + ejes_relativos['ctte_eje_rel_norm_str'].to_numpy() + ")))*"
    factor_normalizador_2 = pd.DataFrame(index=recursos_flujo.index)

    factor_normalizador_2.loc[recursos_flujo['factor_C_2']!='0', 'calcular_str'] = " + (" + recursos_flujo['tipo_f2_num'] + "("\
                                                                                        + ejes_relativos['calc_eje_rel_str'].to_numpy() + ")/"\
                                                                                        + recursos_flujo['tipo_f2_denom'] + "("\
                                                                                        + ejes_relativos['ctte_eje_rel_norm_str'].to_numpy() + "))*"
    factor_normalizador_2.loc[recursos_flujo['factor_C_2']=='0', 'calcular_str'] = ""
    factor_C_2 = pd.DataFrame(index=recursos_flujo.index)
    factor_C_2.loc[recursos_flujo['factor_C_2']!='0', 'calcular_str'] = recursos_flujo['factor_C_2'].astype(str)
    factor_C_2.loc[recursos_flujo['factor_C_2']=='0', 'calcular_str'] = ""
    recursos_flujo['calcular_str'] =  factor_signo['calcular_str'] +\
                                      + recursos_flujo['chr_vector_t'].astype(str) + u"\u1D40(x)["\
                                      + factor_normalizador_1['calcular_str'] + recursos_flujo['factor_C_1']\
                                      + factor_normalizador_2['calcular_str'] + factor_C_2['calcular_str'] + "]"
    return recursos_flujo



def calcular_flujos(regiones, ejes_relativos, mesh_regiones, vectores_valores_eigen,\
                    recursos_v_transpuestos, recursos_flujo, constantes_c,\
                    dimension_mesh=100, n_dimension=100):
    """
    Funcion encargada de construir los flujos.
    Parametros de entrada:
        * regiones: Es un dataframe donde se encuentran almacenados todos los
          valores importantes de las regiones tales como: limites de 'x' y 'y',
          tipo de material(Eps o E0), tipo de funcion ('sin' o 'cos'), etc.
    Salida:
        * flujos: Es un dataframe que contiene el flujo en todas las
          regiones.
    """
    # Se crea el DataFrame de dimension = valores_eigen*n_dimension y se llena con ceros
    flujos = pd.DataFrame(np.zeros(((len(recursos_flujo.index)), dimension_mesh)), index=recursos_flujo.index)
    for n_flujo in flujos.index:
        # Reg.1, Reg.2, ... Reg.n - Iteradores de las regiones
        index_reg_actual = "Reg." + n_flujo.split('V')[1]
        # Se obtiene el vector plano del meshgrid de (y) por cada region
        x_flat = mesh_regiones.loc[index_reg_actual,'x'].to_numpy()
        y_flat = mesh_regiones.loc[index_reg_actual,'y'].to_numpy()
        # Vector eigen pertinente a la region actual
        vector_eig = vectores_valores_eigen.loc[regiones.loc[index_reg_actual, 'chr_eigen']].to_numpy()
        C1 = np.zeros(n_dimension)[:,np.newaxis]
        if recursos_flujo.loc[n_flujo, 'factor_C_1'] != '0':
            C1 = constantes_c[recursos_flujo.loc[n_flujo, 'factor_C_1']].to_numpy()[:,np.newaxis]
        C2 = np.zeros(n_dimension)[:,np.newaxis]
        if recursos_flujo.loc[n_flujo, 'factor_C_2'] != '0':
            C2 = constantes_c[recursos_flujo.loc[n_flujo, 'factor_C_2']].to_numpy()[:,np.newaxis]
        for i in range(0, dimension_mesh):
            # Se obtiene el valor i-esimo del vector aplanado del meshgrid en (x) y en y
            x = x_flat[i]
            y = y_flat[i]
            vector_transpuesto = v_trans.calcular_vector_transpuesto(x,index_reg_actual,vector_eig,recursos_v_transpuestos)
            factor_normalizador_1 = np.diag(np.zeros(n_dimension))
            if recursos_flujo.loc[n_flujo, 'tipo_f1_num'] != '0':
                # Se calcula el numerador del factor normalizador , puede ser:
                # sinh(y-yrel) o cosh(y-yrel)
                func_num_1 = 'np.' + recursos_flujo.loc[n_flujo, 'tipo_f1_num'] + '(vector_eig *('\
                             + ejes_relativos.loc[index_reg_actual, 'calc_eje_rel'] + '))'
                # Se calcula el denominador del factor normalizador , puede ser:
                # dependiendo del flujo en esa region sinh(ctte) o cosh(ctte)
                func_denom_1 = 'np.' + recursos_flujo.loc[n_flujo, 'tipo_f1_denom'] + '(vector_eig *('\
                               + ejes_relativos.loc[index_reg_actual, 'ctte_eje_rel_norm'].astype(str) + '))'
                factor_normalizador_1 = np.diag(eval(func_num_1) / eval(func_denom_1))
            factor_normalizador_2 = np.diag(np.zeros(n_dimension))
            if recursos_flujo.loc[n_flujo, 'tipo_f2_num'] != '0':
                # Se calcula el numerador del factor normalizador , puede ser:
                # sinh(y-yrel) o cosh(y-yrel)
                func_num_2 = 'np.' + recursos_flujo.loc[n_flujo, 'tipo_f2_num'] + '(vector_eig *('\
                             + ejes_relativos.loc[index_reg_actual, 'calc_eje_rel'] + '))'
                # Se calcula el denominador del factor normalizador , puede ser:
                # dependiendo del flujo en esa region sinh(ctte) o cosh(ctte)
                func_denom_2 = 'np.' + recursos_flujo.loc[n_flujo, 'tipo_f2_denom'] + '(vector_eig *('\
                               + ejes_relativos.loc[index_reg_actual, 'ctte_eje_rel_norm'].astype(str) + '))'
                factor_normalizador_2 = np.diag(eval(func_num_2) / eval(func_denom_2))
            # Se calcula el factor final
            final_factor = (vector_transpuesto @ np.diag(vector_eig) ) @ (factor_normalizador_1 @ C1 + factor_normalizador_2 @ C2)
            # finalmente se agrega al dataframe y se le agrega el signo evaluando con el diccionario operator
            # operator['+'](0,final_factor)  || operator['-'](0,final_factor)
            if recursos_flujo.loc[n_flujo, 'signo'] == '-':
                flujos.loc[n_flujo, i] = -final_factor
            else:
                flujos.loc[n_flujo, i] = final_factor
    # Para evitar confusiones los indices de las columnas van desde 1 en adelante
    flujos.columns += 1
    return flujos
