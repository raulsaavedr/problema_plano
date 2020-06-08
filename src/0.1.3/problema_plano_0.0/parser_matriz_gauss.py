import numpy as np
import pandas as pd
import csv
import re
from string import ascii_uppercase

from config import conf


def cargar_matriz_coeficientes(n_dimension):
    matriz_coeficientes_list = []
    n_dimension_matriz_gauss = conf.data['vars']['n_dimension_matriz_gauss']
    filename = 'csv/' + conf.data['env']['path'] + '/matriz_coeficientes.csv'
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        # Un for a cada fila del archivo
        for row in spamreader:
            expression_row_list = []
            # Un for a cada expresion contenida en cada fila, ejemplo Na1=k1*N*D1
            for expression in row:
                # Separe la expresion actual en una lista para poder extraer variable por variable
                expr_mod = expression.split('=')[1].split('*')
                # De la lista donde residen las variables limpiela de espacios vacios 
                expr_mod = [i for ext in [list(filter(None, termino.split(' '))) for termino in expr_mod] for i in ext]
                # Un for a cada variable contenida en la expresion ejemplo: k1
                local_expression = ''
                for variable_calculo in expr_mod:
                    # Busque signos negativos
                    if re.search('^-', variable_calculo):
                        # Si la variable_calculo inicia con un menos se agrega un menos a la expresion local
                        local_expression += '-'
                        # Se le quita el menos a variable_calculo para que pueda encontrar las siguientes expresiones
                        variable_calculo = variable_calculo.split('-')[1]
                    # Busque epsilon cero
                    if re.search('^E0', variable_calculo):
                        # Encuentre todas los epsilon que sean E0
                        epsilons_finded = re.findall('^E0', variable_calculo)
                        local_expression += '1*'
                    # Busque epsilon relativo
                    if re.search('^Er[1-9]+', variable_calculo):
                        # Encuentre todos los epsilon Er y extraiga solamente el numero
                        epsilons_finded = re.findall('^Er([1-9.]+)', variable_calculo)
                        # Sumele uno al numero extraido
                        local_expression += str(int(epsilons_finded[0]) + 1)  + '*'
                    # Busque matriz cuadrada de ceros
                    if re.search('^C[^A-Z0-9]*$', variable_calculo):
                        # Encuentre todas las palabras que inicien por C y que este seguido de letras minusculas (Cero)
                        constantes_k_finded = re.findall('^C[^A-Z0-9]*$', variable_calculo)
                        local_expression += "np.diag(np.zeros(" + str(n_dimension) + "))*"
                    # Busque matriz diagonal de unos
                    if re.search('^Uno*$', variable_calculo):
                        # Encuentre todas las palabras que inicien por U y que este seguido de letras minusculas (Uno)
                        constantes_k_finded = re.findall('^U[^A-Z0-9]*$', variable_calculo)
                        local_expression += "np.eye(" + str(n_dimension) + ")*"
                    # Busque las constantes k
                    if re.search('^k\d+', variable_calculo):
                        # Encuentre todas las k que tenga al menos un numero esto es: k1,k2,k3,k4...
                        constantes_k_finded = re.findall('^k\d+', variable_calculo)
                        local_expression += "constantes_k.loc['" + constantes_k_finded[0] + "' ,'calcular']*"
                    # Busque vectores eigen
                    if re.search('^[A-Z][^a-zA-Z0-9]*$', variable_calculo):
                        # Encuentre todas las letras mayusculas de un solo caracter
                        vectores_eigen_finded = re.findall('^[A-Z][^a-zA-Z0-9]*$', variable_calculo)
                        local_expression += "matrices_diagonales_valores_eig.loc['" + vectores_eigen_finded[0] + "'].to_numpy()*"
                    # Busque matrices diagonales hiperbolicas
                    if re.search('^D+[0-9]+', variable_calculo):
                        # Encuentre todas la
                        diagonales_hiperbolicas_finded = re.findall('^D+[0-9]+', variable_calculo)
                        local_expression += "matrices_diagonales_hiperbolicas.loc['" + diagonales_hiperbolicas_finded[0] + "'].to_numpy()*"
                    # Busque Matrices de acoplamiento
                    if re.search('^M+[0-9]+$', variable_calculo):
                        # Encuentre todas las letras que empiecen con M mayuscula seguido de uno o mas numeros
                        matrices_acopladoras_finded = re.findall('^M+[0-9]+', variable_calculo)
                        local_expression += "matrices_acoplamiento.loc['" + matrices_acopladoras_finded[0] + "'].to_numpy()*"
                    # Busque Matrices transpuestas
                    if re.search('^M+[0-9]+T', variable_calculo):
                        # Encuentre todas las letras que empiecen con M mayuscula seguido de uno o mas numeros seguido de una T
                        matrices_transpuestas_acopladoras_finded = re.findall('^(M+[0-9]+)T', variable_calculo)
                        local_expression += "matrices_acoplamiento_trans.loc['" + matrices_transpuestas_acopladoras_finded[0] + "'].to_numpy()*"
                # Se elemina el ultimo caracter de la expresion local que es un asterisco
                local_expression = local_expression[:-1]
                # Agregue cada expresion local a una lista que almacena todas las expresiones por fila
                expression_row_list.append(local_expression)
            # Luego agregue cada fila a la lista matriz_coeficientes_list
            matriz_coeficientes_list.append(expression_row_list)
    index = [ascii_uppercase[i] for i in range(0, n_dimension_matriz_gauss)]
    columns = ['C' + str(i) for i in range(1, n_dimension_matriz_gauss + 1)]
    matriz_coeficientes_calcular_str = pd.DataFrame(matriz_coeficientes_list, index=index, columns=columns)
    return matriz_coeficientes_calcular_str


def cargar_variable_valores_dependientes(n_dimension):
    vectores_valores_dependientes_list = []
    n_dimension_matriz_gauss = conf.data['vars']['n_dimension_matriz_gauss']
    filename = 'csv/' + conf.data['env']['path'] + '/vectores_terminos_dependientes.csv'
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        # Un for para cada fila
        for row in spamreader:
            local_expression = ''
            expression = row[0]
            expr_mod = expression.split('=')[1].split(' ')
            for variable_calculo in expr_mod:
                # Busque signos negativos
                if re.search('^-', variable_calculo):
                    # Si la variable_calculo inicia con un menos se agrega un menos a la expresion local
                    local_expression += '-'
                if re.search('^\+', variable_calculo):
                    # Si la variable_calculo inicia con un mas se agrega un mas a la expresion local
                    local_expression += '+'
                # Busque vectores distorsionadores
                if re.search('^S[0-9]+', variable_calculo):
                    # Encuentre todas las palabras que inicien por S y que este seguido de numeros
                    vector_distorsionador_finded = re.findall('^S[0-9]+', variable_calculo)
                    local_expression += "vectores_distorsionadores.loc['" + vector_distorsionador_finded[0] + "'].to_numpy()"
                #Busque vector transpuesto de ceros
                if re.search('^CeroV', variable_calculo):
                    # Encuentre todas las palabras que inicien por C y que este seguido de letras minusculas (Cero)
                    constantes_k_finded = re.findall('^CeroV', variable_calculo)
                    local_expression += "np.zeros(" + str(n_dimension) + ")"
            vectores_valores_dependientes_list.append(local_expression)
    index = [ascii_uppercase[i] for i in range(0, n_dimension_matriz_gauss)]
    vectores_valores_dependientes_calcular_str = pd.Series(vectores_valores_dependientes_list, index=index, dtype='object')
    return vectores_valores_dependientes_calcular_str


def main():
    conf.load(problema_plano='ejemplo')
    matriz_coeficientes_calcular_str = cargar_matriz_coeficientes(n_dimension=100)
    vectores_valores_dependientes_calcular_str = cargar_variable_valores_dependientes(n_dimension=100)
    print(matriz_coeficientes_calcular_str.loc[:,['C1','C2']].to_string())
    print(vectores_valores_dependientes_calcular_str.loc['G'])
    print(vectores_valores_dependientes_calcular_str.loc['F'])
    print(vectores_valores_dependientes_calcular_str.loc['D'])

if __name__ == '__main__':
    main()
