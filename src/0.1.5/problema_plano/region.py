import pandas as pd
import numpy as np

from config import conf
__doc__="""
Este modulo es equivalente a: f02_region(n)
    En este archivo se crearan funciones para la carga
    en memoria del fichero region.csv
    Este fichero contiene toda la informacion respectiva
    a cada region.
"""

def cargar_regiones(n_dimension=100):
    """
    Funcion encargada de cargar el archivo regiones.csv en un DataFrame
    Parametros de entrada:
        * n_dimension: Nro. de dimensiones de los vectores, matrices a utilizar
          dentro de la resolucion del problema plano

    Salida:
        retorna un DataFrame con base a lo que esta en dicho archivo
        Contiene por cada region del problema plano:
            * xi: x inicial
            * xf: x final
            * yi: y inicial
            * yf: y final
            * Eps: Material (E0, Er)
            * chr_eigen: Con que valor eigen se calcula la region
            * f_eigen: Si el valor eigen es tipo seno o coseno
            * Rango de x: rango de x en variable tipo string
            * Rango de y: rango de y en variable tipo string
            * n_dimension: numero de dimension de los vectores y matrices del
              problema plano

    Nota: Para un ejemplo dado remitirse a la funcion main de este archivo.

    Ejemplo de salida:

    """
    # Se hace lectura del archivo y se almacena como DataFrame
    filename = 'csv/' + conf.data['env']['path'] + '/regiones.csv'
    regiones = pd.read_csv(filename)
    # Se cambia el index de: 0,...,N por este otro: 1,...,N
    regiones.index += 1
    # Cambiar el nombre de los index de numeros a: Region 1, Region 2,..., Region n
    regiones.index = ['Reg.' + str(i) for i in regiones.index]
    # Agregar rango de 'x' y de 'y' en forma de cadena
    regiones['rango_de_x'] = (regiones['xi'].astype(str) + "<=x<=" + regiones['xf'].astype(str))
    regiones['rango_de_y'] = (regiones['yi'].astype(str) + "<=y<=" + regiones['yf'].astype(str))
    regiones['n_dimension'] = int(n_dimension)
    return regiones

def cargar_ejes_relativos(regiones):
    ejes_relativos = pd.DataFrame()
    # Se definen los ejes relativos
    # Si el flujo sube se define el eje relativo de y como y - yn
    ejes_relativos.loc[regiones['direcc_de_flujo']=='subiendo','calc_eje_rel'] =  "y - " + regiones['yf'].astype(str)
    # Si el flujo baja se define el eje relativo de y como yn - y
    ejes_relativos.loc[regiones['direcc_de_flujo']=='bajando','calc_eje_rel'] =  regiones['yf'].astype(str) + " - y "
    # Si el flujo va en ambos sentidos se define el eje relativo de y como y - (yn+ym)/2
    ejes_relativos.loc[regiones['direcc_de_flujo']=='ambos sentidos','calc_eje_rel'] = "y - " + ((regiones['yi'] + regiones['yf']) / 2).astype(str)
    # Si hay alguna region que inicie desde cero entonces en esa region no hay eje relativos
    ejes_relativos.loc[regiones['yi']== 0,'calc_eje_rel'] = 'y'
    # Se define lo mismo pero ahora de forma string
    # Si el flujo sube se define el eje relativo de y como y - yn
    ejes_relativos.loc[regiones['direcc_de_flujo']=='subiendo','calc_eje_rel_str'] =  "y - y" + regiones['yf'].astype(str)
    # Si el flujo baja se define el eje relativo de y como yn - y
    ejes_relativos.loc[regiones['direcc_de_flujo']=='bajando','calc_eje_rel_str'] =  "y" + regiones['yf'].astype(str) + " - y "
    # Si el flujo va en ambos sentidos se define el eje relativo de y como y - (yn+ym)/2
    ejes_relativos.loc[regiones['direcc_de_flujo']=='ambos sentidos','calc_eje_rel_str'] = "y - ((y" + regiones['yi'].astype(str)\
                                                                                          + " + y" + regiones['yf'].astype(str) + ")/2)"
    # Si hay alguna region que inicie desde cero entonces en esa region no hay eje relativos
    ejes_relativos.loc[regiones['yi']== 0,'calc_eje_rel_str'] = 'y'

    # Se definen las constantes para normalizacion
    # Si el flujo sube o baja se define la constante para normalizacion como: yfinal - yinit
    ejes_relativos.loc[regiones['direcc_de_flujo']=='subiendo','ctte_eje_rel_norm'] =  (regiones['yf'] - regiones['yi'])
    ejes_relativos.loc[regiones['direcc_de_flujo']=='bajando','ctte_eje_rel_norm'] =  (regiones['yf'] - regiones['yi'])
    # Si el flujo va en ambos sentidos se define la constante para normalizacion como: (yfinal - yinit )/2
    ejes_relativos.loc[regiones['direcc_de_flujo']=='ambos sentidos','ctte_eje_rel_norm'] = ((regiones['yf'] - regiones['yi']) / 2)
    # Se definen las constantes para normalizacion pero en formato cadena
    # Si el flujo sube o baja se define la constante para normalizacion como: yfinal - yinit
    ejes_relativos.loc[regiones['direcc_de_flujo']=='subiendo','ctte_eje_rel_norm_str'] =  "y" + regiones['yf'].astype(str) + " - y" + regiones['yi'].astype(str)
    ejes_relativos.loc[regiones['direcc_de_flujo']=='bajando','ctte_eje_rel_norm_str'] =  "y" + regiones['yf'].astype(str) + " - y" + regiones['yi'].astype(str)
    # Si el flujo va en ambos sentidos se define la constante para normalizacion como: (yfinal - yinit )/2
    ejes_relativos.loc[regiones['direcc_de_flujo']=='ambos sentidos','ctte_eje_rel_norm_str'] = "((y" + regiones['yf'].astype(str) + " - y" + regiones['yi'].astype(str) + ")/2)"
    return ejes_relativos

def cargar_mesh_regiones(regiones, delta=10):
    """
    Funcion encargada de hacer un mesh para cada region
    """
    dimension_mesh = delta*delta
    index = pd.MultiIndex.from_tuples([(reg, i) for reg in regiones.index for i in ['x','y']])
    columns = [i for i in range(1,dimension_mesh + 1)]
    ndarray = np.zeros(((len(regiones.index) * 2), dimension_mesh))
    mesh_regiones = pd.DataFrame(data=ndarray, index=index, columns=columns)
    for reg in regiones.index:
        x_axis_p_plano = np.linspace(regiones.loc[reg,'xi'],regiones.loc[reg,'xf'], delta)
        y_axis_p_plano = np.linspace(regiones.loc[reg,'yi'],regiones.loc[reg,'yf'], delta)
        x, y = np.meshgrid(x_axis_p_plano, y_axis_p_plano)
        x = x.reshape(1,dimension_mesh)
        y = y.reshape(1,dimension_mesh)
        mesh_regiones.loc[reg, 'x'] = x[0]
        mesh_regiones.loc[reg, 'y'] = y[0]
    return mesh_regiones, dimension_mesh
