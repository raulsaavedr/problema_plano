import pandas as pd

from config import conf

__doc__ = """
Modulo encargado de definir algunas funciones para el manejo de las constantes
k utilizadas en el calculo (k1, k2, ..., kn)
Es equivalente a una parte del principio de f09_Matriz_de_Gauss()
"""

def cargar_constantes_k():
    """
    Funcion que carga en un dataframe lo necesario para construir como
    las constantes que seran utilizadas en el calculo del problema plano.
    El archivo constantes_k_calculo.csv debe ser definido de la siguiente
    manera:
        La primera fila se define como la fila de los nombres que tomara cada
    columna, esta fila luce asi:
                xi,xf,reciproco

        Donde:
            * xi: desplazamiento inicial en x
            * xf: desplazamiento final en x
            * reciproco: si es necesario definir el reciproco de una constante
              ya creada

        La formula de las constantes se construye de la siguiente manera:
            reciproco = no   -------->   kn = 2 / (xf-xi)
            reciproco = si   -------->   kn = (xf-xi) / 2
    """
    filename = 'csv/' + conf.data['env']['path'] + '/constantes_k.csv'
    constantes_k_calculo = pd.read_csv(filename)
    # Se arregla el index para que sea k1,k2,...., kn
    constantes_k_calculo.index = ['k'+ str(i + 1) for i in range(len(constantes_k_calculo))]
    # Se agrega el calcular en punto flotante
    constantes_k_calculo.loc[constantes_k_calculo['reciproco']  == 'no', 'calcular'] = 2 / (constantes_k_calculo['xf'] - constantes_k_calculo['xi'] )
    constantes_k_calculo.loc[constantes_k_calculo['reciproco']  == 'si', 'calcular'] = (constantes_k_calculo['xf'] - constantes_k_calculo['xi'] ) / 2
    # Se agrega el calcular en cadena
    constantes_k_calculo.loc[constantes_k_calculo['reciproco']  == 'no', 'calcular_str'] = '2/(x' + constantes_k_calculo['xf'].astype(str) + '-x'\
                                                                                       + constantes_k_calculo['xi'].astype(str) + ')'
    constantes_k_calculo.loc[constantes_k_calculo['reciproco']  == 'si', 'calcular_str'] = '(x' + constantes_k_calculo['xf'].astype(str) + '-x'\
                                                                                       + constantes_k_calculo['xi'].astype(str) + ')/2'
    return constantes_k_calculo


def main():
    config.init(problema_plano='ejemplo')
    constantes_k = cargar_constantes_k()
    print(constantes_k)


if __name__ == '__main__':
    main()
