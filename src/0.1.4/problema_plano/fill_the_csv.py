import csv
import logging
import yaml

from config import conf

logger = logging.getLogger('fill_the_csv')
logger.setLevel(conf.get_logging_level('fill_the_csv'))
logger.disabled = conf.data['debugging']['fill_the_csv']['disabled']

__doc__ = """
Modulo hecho para llenar de manera segura y ordenada los archivos csv necesarios
"""

def load_artificio_potencial():
    """\nload_artificio_potencial():\n
    Funcion utilizada para llenar el archivo csv: artificio_potencial.csv. Se
    debe llenar de la siguiente manera:
        * sum. 1er termino: Suma de primer termino debe ser un numero entero,
                            cero(0) o uno(1)
        * term. (x-ca/cb-ca): Es una condicion, si este termino se encuentra en
                              el potencial (0)no (1)si
        * ca: desplazamiento en x del numerador, y termino de resta en el deno-
               minador.
        * cb: primer termino en la resta del denominador
        * chr_vector_t: el nombre del vector transpuesto relacionado con ese po-
                        tencial. Debe ser una letra [A-Z]
        * tipo_f1: Tipo de funcion trigonometrica del primer factor puede ser:
                   (1)sinh o (2)cosh
        * termino_C_1: La constante Cn que multiplica el primer termino debe ser
                      una sola (C1 o C2 o C3 o .... Cn)
        * tipo_f2: Tipo de funcion trigonometrica del segundo factor puede ser:
                   (1)sinh o (2)cosh. Si no existe entonces debe ser 0.
        * termino_C_2: La constante Cn que multiplica el segundo factor debe ser
                      una sola (C1 o C2 o C3 o .... Cn). Si no existe entonces
                      debe ser 0.

    La expresion del potencial n es construida de la siguiente forma:
    Vn(x,y)=sum. 1er termino - (x-ca/cb-ca) + chr_vector_t(...
            [tipo_f1(chr_eigen*term.y)/tipo_f1(chr_eigen*term.y)]*termino_C_1...
            + [tipo_f2(chr_eigen*term.y)/tipo_f2(chr_eigen*term.y)]*termino_C_2)

    Nota: Es importante destacar que el programa solo utiliza este archivo para
    calcular los dos artificios tanto de potencial como de flujo.

    Nota 2: Los valores de (chr_eigen) y (term.y) son configurados automatica-
    mente, gracias a regiones.csv por ende no se piden estos valores para este
    archivo.
    """
    print(load_artificio_potencial.__doc__)

    lista = [
        ['sum. 1er termino', 'term. (x-ca/cb-ca)', 'ca', 'cb', 'chr_vector_t',
         'tipo_f1', 'tipo_f2', 'factor_C_1', 'factor_C_2']
    ]
    n_potenciales = conf.data['vars']['n_regiones']
    tp_map = {0:'0', 1:'sinh', 2:'cosh'}
    term_map = {0:'no', 1:'si'}
    for i in range(n_potenciales):
        ca = 0
        cb = 0
        print(f"\n\tPotencial V{i+1}(x,y)")
        termino_sum = input("Inserte sum. 1er termino (1 o 0):")
        int_term_fraccion = int(input("Existe el termino (x-ca/cb-ca) (0)no (1)si:"))
        if int_term_fraccion == 1:
            ca = input("Inserte ca (numero):")
            cb = input("Inserte cb (numero):")
        chr_vector_t = input("Inserte el nombre del vector transpuesto:").upper()
        int_tp_1 = int(input("Inserte el tipo de funcion trigonometrica del primer factor (1)sinh (2)cosh:"))
        factor_C_1 = input("Inserte la constante que multiplica al primer factor (Cn):").upper()
        int_tp_2 = int(input("Inserte el tipo de funcion trigonometrica del segundo factor (0)no existe (1)sinh (2)cosh:"))
        factor_C_2 = input("Inserte la constante que multiplica al segundo factor (0)no existe o (Cn):").upper()
        term_x_ca_cb = f"-(x-x{ca}/x{cb}-x{ca})" if int_term_fraccion == 1 else ""
        if int_tp_2 == 0:
            print(f"V{i+1}(x,y)={termino_sum}{term_x_ca_cb}+{chr_vector_t}\u1D40(" + \
                  f"[{tp_map[int_tp_1]}(chr_eigen*term.y)/{tp_map[int_tp_1]}(chr_eigen*term.y)]*" +\
                  f"{factor_C_1})")
        else:
            print(f"V{i+1}(x,y)={termino_sum}{term_x_ca_cb}+{chr_vector_t}\u1D40(" + \
                  f"[{tp_map[int_tp_1]}(chr_eigen*(term.y))/{tp_map[int_tp_1]}(chr_eigen*term.y)]*" +\
                  f"{factor_C_1} + [{tp_map[int_tp_2]}(chr_eigen*term.y)/{tp_map[int_tp_2]}" +\
                  f"(chr_eigen*term.y)]*{factor_C_2})")
        lista.append([termino_sum, term_map[int_term_fraccion], ca, cb, chr_vector_t,
                      tp_map[int_tp_1], tp_map[int_tp_2], factor_C_1, factor_C_2])

    filename = 'csv/' + conf.data['env']['path'] + '/artificio_potencial.csv'
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(lista)
    # Se aplican los cambios al diccionario de configuracion data
    conf.data['csv_filled']['artificio_potencial'] = True
    conf.update_config_file()
    print(f"El archivo {filename} ha sido creado con exito!")

def load_coeficientes_no_cero_matriz_coeficientes():
    """\nload_coeficientes_no_cero_matriz_coeficientes():\n
    Funcion que sirve para llenar los coeficientes que no son |0|, es decir,
    que no son matriz cuadrada de ceros. Se solicita que se llene por filas con
    los coeficientes respectivos de cada fila, separados por coma. Ejemplo:
        Ingrese fila 1: C1,C2,C3
        Ingrese fila 2: C2,C4,C7,C8
        ...
        Ingrese fila n_dimension_matriz_gauss: C8,C9,Cn_dimension_matriz_gauss
    """
    print(load_coeficientes_no_cero_matriz_coeficientes.__doc__)
    n_dimension_matriz_gauss = int(input("Inserte la dimension de la matriz de gauss:"))
    lista = []
    for i in range(n_dimension_matriz_gauss):
        lista.append(input(f"Inserte fila {i+1}:").split(','))
    filename = 'csv/' + conf.data['env']['path'] + \
               '/coeficientes_no_cero_matriz_coeficientes.csv'
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(lista)
    # Se aplican los cambios al diccionario de configuracion data
    conf.data['csv_filled']['coeficientes_no_cero_matriz_coeficientes'] = True
    conf.data['vars']['n_dimension_matriz_gauss'] = n_dimension_matriz_gauss
    conf.update_config_file()
    print(f"El archivo {filename} ha sido creado con exito!")

def load_constantes_k():
    """\nload_constantes_k():\n
    Funcion utilizada para llenar el archivo csv: constantes_k.csv
    Se procede a llenar el archivo con los siguientes valores:
        * x inicial: Debe ser entero
        * x final: Debe ser entero
        * reciproco: Debe ser entero (1)Si o (2)No

        La constantes de calculo kn son construidas de la siguiente manera:
            reciproco = no   -------->   kn = 2 / (xf-xi)
            reciproco = si   -------->   kn = (xf-xi) / 2
    """
    print(load_constantes_k.__doc__)
    lista = [['xi', 'xf', 'reciproco']]
    reciproco_lamb = {1:'si', 2:'no'}
    n_constantes_k = int(input("Inserte la cantidad de constantes kn:"))
    for i in range(n_constantes_k):
        print(f"\n\t\tConstante k{i+1}")
        xf = input("Inserte x final:")
        xi = input("Inserte x inicial:")
        int_reciproco = int(input("Inserte si la constante es reciproca (1)Si (2)No:"))
        if reciproco_lamb[int_reciproco] == 'si':
            print(f"k{i+1}=(x{xf}-x{xi})/2")
        else:
            print(f"k{i+1}=2/(x{xf}-x{xi})")
        lista.append([xi, xf, reciproco_lamb[int_reciproco]])

    filename = 'csv/' + conf.data['env']['path'] + '/constantes_k.csv'
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(lista)
    # Se aplican los cambios al diccionario de configuracion data
    conf.data['csv_filled']['constantes_k'] = True
    conf.data['vars']['n_constantes_k'] = n_constantes_k
    conf.update_config_file()
    print(f"El archivo {filename} ha sido creado con exito!")

def load_funciones_hiperbolicas():
    """\nload_funciones_hiperbolicas():\n
    Funcion utilizada para llenar el archivo csv: funciones_hiperbolicas.csv
    Debe estar definido para cada una de las matrices diagonales de funciones
    hiperbolicas y contiene lo siguiente por cada matriz:
        * chr_eigen: El nombre del vector eigen debe ser una letra [A-Z]
        * y inicial: Debe ser un valor entero
        * y final: Debe ser un valor entero
        * tipo: funcion trigonometrica (1)tanh o (2)coth. Debe ser entero
        * divisor: puede ser 1 o 2 depende de cada matriz. Debe ser entero

        La expresion es contruida de la siguiente forma:
            Di = tipo(chr_eigen*(yf-yi)/divisor)
    """
    print(load_funciones_hiperbolicas.__doc__)
    lista = [['chr_eigen', 'yi', 'yf', 'tipo', 'divisor']]
    n_diagonales_hiperbolicas = int(input("Inserte el numero de matrices diagonales de func.hiperbolicas:"))
    tipo_lamb = {1:'tanh', 2:'coth'}
    for i in range(n_diagonales_hiperbolicas):
        print(f"\n\t\tDiagonal D{i+1}")
        int_tipo = int(input("Inserte la funcion trigonometrica de la diagonal (1)tanh (2)coth:"))
        chr_eigen = input("Inserte el nombre del vector eigen:")
        yf = input("Inserte y final:")
        yi = input("Inserte y inicial:")
        divisor = input("Inserte si es divido por (1) o por (2):")
        print(f"D{i+1}={tipo_lamb[int_tipo]}({chr_eigen}*(y{yf}-y{yi})/{divisor})")
        lista.append([chr_eigen, yi, yf, tipo_lamb[int_tipo], divisor])

    filename = 'csv/' + conf.data['env']['path'] + '/funciones_hiperbolicas.csv'
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(lista)
    # Se aplican los cambios al diccionario de configuracion data
    conf.data['csv_filled']['funciones_hiperbolicas'] = True
    conf.data['vars']['n_diagonales_hiperbolicas'] = n_diagonales_hiperbolicas
    conf.update_config_file()
    print(f"El archivo {filename} ha sido creado con exito!")

def load_matrices_cuadradas_acoplamiento():
    """\nload_matrices_cuadradas_acoplamiento():\n
    Funcion utilizada para llenar el archivo csv: matrices_cuadradas_acopla-
    miento.csv. Se debe llenar de la siguiente manera:
        * xi : limite de integracion inicial. (numero entero)
        * xf: limite de integracion final.(numero entero)
        * tipo_1: tipo de funcion del primer factor (1)sin (2)cos. Debe ser un
                  numero entero
        * ch_eigen_1: es el vector eigen con que se calculara el primer
          factor. Letra [A-Z]
        * desp_1: desplazamiento en x del primer factor.(numero entero)
        * tipo_2: tipo de funcion del segundo factor (1)sin (2)cos. Debe ser un
                  numero entero
        * ch_eigen_2: es el vector eigen con que se calculara el segundo
          factor.Letra [A-Z]
        * desp_2: desplazamiento en x del segundo factor.(numero entero)
    """
    print(load_matrices_cuadradas_acoplamiento.__doc__)
    lista = [['xi', 'xf', 'tipo_1', 'chr_eigen_1', 'desp_1', 'tipo_2', 'chr_eigen_2', 'desp_2']]
    n_matrices_acopladoras = int(input("Inserte el numero de matrices acopladoras:"))
    tp_lamb = {1:'sin', 2:'cos'}
    for i in range(n_matrices_acopladoras):
        print(f"\n\tMatriz de acoplamiento M{i+1}")
        xi = input("Inserte limite de integracion inicial:")
        xf = input("Inserte limite de integracion final:")
        int_tp_1 = int(input("Inserte el tipo de funcion trigonometrica del primer factor (1)sin (2)cos:"))
        chr_eigen_1 = input("Inserte el nombre del vector eigen del primer factor:").upper()
        desp_1 = input("Inserte desplazamiento en x del primer factor:")
        int_tp_2 = int(input("Inserte el tipo de funcion trigonometrica del segundo factor (1)sin (2)cos:"))
        chr_eigen_2 = input("Inserte el nombre del vector eigen del segundo factor:").upper()
        desp_2 = input("Inserte desplazamiento en x del segundo factor:")
        print(f"M{i+1}(m,n)=\u222B({tp_lamb[int_tp_1]}({chr_eigen_1}m*(x-x{desp_1}))*{tp_lamb[int_tp_2]}" +\
                 f"({chr_eigen_2}n*(x-x{desp_2}))) de x{xi} a x{xf}")
        lista.append([xi, xf, tp_lamb[int_tp_1], chr_eigen_1, desp_1, tp_lamb[int_tp_2], chr_eigen_2, desp_2])

    filename = 'csv/' + conf.data['env']['path'] + '/matrices_cuadradas_acoplamiento.csv'
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(lista)
    # Se aplican los cambios al diccionario de configuracion data
    conf.data['csv_filled']['matrices_cuadradas_acoplamiento'] = True
    conf.data['vars']['n_matrices_acopladoras'] = n_matrices_acopladoras
    conf.update_config_file()
    print(f"El archivo {filename} ha sido creado con exito")

def load_matriz_coeficientes():
    """\nload_matriz_coeficientes():\n
    Funcion utilizada para llenar el archivo csv: matriz_coeficientes.csv
    La referencia acontinuacion debe ser utilizada, tener en cuenta que es,
    case-sensitive (respetar mayusculas y minusculas).
        * Material Epsilon: E0, Er1, Er2, Er3, Ern
        * Constantes k: k1, k2, kn
        * Matriz diagonal de unos: Uno
        * Matriz cuadradaa de ceros: Cero
        * Vectores Eigen: A, B, C, ..., Z
        * Matrices diagonales hiperbolicas: D1, D2, ..., Dn
        * Matrices de acoplamiento:  M1, M2, ..., Mn
        * Matrices transpuestas: M1T, M2T, ..., MnT
    """
    try:
        filename = 'csv/' + conf.data['env']['path']  + '/coeficientes_no_cero_matriz_coeficientes.csv'
        lista_coeficientes = list(csv.reader(open(filename)))
    except Exception as e:
        # File not found error
        load_coeficientes_no_cero_matriz_coeficientes()
    # Se convierte en lista de enteros [[1,2,3],...,[4,...,n_dimension_matriz_gauss]]
    coeficientes_leer = [[int(element[1:]) for element in row_list] for row_list in lista_coeficientes]
    logger.debug(coeficientes_leer)
    print(load_matriz_coeficientes.__doc__)
    letras = [chr(char) for char in range(ord('a'), ord('a') + len(coeficientes_leer))]
    matriz_coeficientes = []
    for letra in letras:
        matriz_coeficientes.append(['N' + letra + str(i+1) + '=Cero' for i in range(len(letras))])

    for i,fila in enumerate(matriz_coeficientes):
        for x in coeficientes_leer[i]:
            aux = fila[x-1].split('C')[0]
            fila[x-1]= aux + input(aux)
            print(f"Ingreso: {fila[x-1]}")

    filename = 'csv/' + conf.data['env']['path'] + '/matriz_coeficientes.csv'
    logger.info(filename)
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(matriz_coeficientes)
    conf.data['csv_filled']['matriz_coeficientes'] = True
    conf.update_config_file()
    print(f"El archivo {filename} ha sido creado con exito!")

def load_regiones():
    """\nload_regiones():\n
    Funcion utilizada para llenar el archivo csv: regiones.csv
    Se pide por cada region lo siguiente:
        * x inicial: debe ser un valor entero
        * x final: debe ser un valor entero
        * y inicial: debe ser un valor entero
        * y final: debe ser un valor entero
        * Eps: E0, Er, Er1, Er2, etc..
        * chr_eigen: debe ser una Letra [A - Z]
        * f_eigen: Si el valor eigen es tipo seno (sin) o coseno (cos)
        * direcc_de_flujo: debe ser un valor entero
    Nota: Los ejes de entrada de region son absolutos(No dependen del eje rela-
    tivo fijado por la region)
    """
    print(load_regiones.__doc__)
    lista = [['xi', 'xf', 'yi', 'yf', 'eps', 'chr_eigen', 'f_eigen', 'direcc_de_flujo']]
    n_regiones = int(input("Inserte el numero de regiones:"))
    flujo_lamb = {1:'subiendo', 2:'bajando', 3: 'ambos sentidos'}
    f_eigen_lamb = {1:'sin', 2:'cos'}
    for i in range(n_regiones):
        print(f"\n\t\tRegion.{i+1}")
        xi = input("Inserte x inicial:")
        xf = input("Inserte x final:")
        yi = input("Inserte y inicial:")
        yf = input("Inserte y final:")
        eps = input("Inserte material (Eps):")
        chr_eigen = input("Inserte el nombre del vector eigen:")
        int_f_eigen = int(input("Inserte la funcion trigonometricas del vector eigen (1)sin (2)cos:"))
        int_dir_flujo = int(input("Inserte la direccion de flujo (1)Subiendo, (2)Bajando, (3)Ambos sentidos:"))
        lista.append([xi, xf, yi, yf, eps, chr_eigen, f_eigen_lamb[int_f_eigen], flujo_lamb[int_dir_flujo]])

    filename = 'csv/' + conf.data['env']['path'] + '/regiones.csv'
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(lista)
    # Se aplican los cambios al diccionario de configuracion data
    conf.data['csv_filled']['regiones'] = True
    conf.data['vars']['n_regiones'] = n_regiones
    conf.update_config_file()
    print(f"El archivo {filename} ha sido creado con exito!")


def load_vectores_distorsionadores():
    """\nload_vectores_distorsionadores():\n
    Funcion utilizada para llenar el archivo csv: regiones.csv
    Se pide por cada region lo siguiente:
        * xi: limite de integracion inicial. (numero entero)
        * xf: limite de integracion final.(numero entero)
        * tipo: tipo de funcion trigonometrica (1)sin (2)cos. Debe ser un
                numero entero
        * ch_eigen: es el vector eigen con que se calculara la integral.
        * por_partes: indica si la integral incluye el termino (x-xi)/(x-xf)
                      (1) si, (2)no
    """
    print(load_vectores_distorsionadores.__doc__)
    lista = [['xi', 'xf', 'tipo', 'por_partes', 'ca', 'cb', 'chr_eigen']]
    n_vectores_distorsionadores = int(input("Inserte el numero de vectores distorsionadores:"))
    tipo_map = {1:'sin', 2:'cos'}
    decision_map = {1:'si', 2:'no'}
    for i in range(n_vectores_distorsionadores):
        print(f"\n\t\tVector distorsionador S{i+1}")
        xi = input("Inserte limite de integracion inicial:")
        xf = input("Inserte limite de integracion final:")
        int_por_partes = int(input("Inserte si tiene termino (x-xf)/(xf-xi)  (1)si (2)no:"))
        if int_por_partes == 1:
            ca = xi
            cb = xf
        else:
            ca = 0
            cb = 0
        int_tp = int(input("Inserte el tipo de funcion trigonometrica (1)sin (2)cos:"))
        chr_eigen = input("Inserte el nombre del vector eigen:").upper()
        if int_por_partes == 1:
            print(f"S{i+1}(m)=\u222B((x-x{xi})/(x{xf}-x{xi}))*{tipo_map[int_tp]}({chr_eigen}m*x) de x{xi} a x{xf}")
        else:
            print(f"S{i+1}(m)=\u222B{tipo_map[int_tp]}({chr_eigen}m*x) de x{xi} a x{xf}")
        lista.append([xi, xf, tipo_map[int_tp],  decision_map[int_por_partes],
                                ca, cb, chr_eigen])

    filename = 'csv/' + conf.data['env']['path'] + '/vectores_distorsionadores.csv'
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(lista)
    # Se aplican los cambios al diccionario de configuracion data
    conf.data['csv_filled']['vectores_distorsionadores'] = True
    conf.data['vars']['n_vectores_distorsionadores'] = n_vectores_distorsionadores
    conf.update_config_file()
    print(f"El archivo {filename} ha sido creado con exito!")

def load_vectores_terminos_dependientes():
    """\nload_vectores_terminos_dependientes():\n
    Funcion utilizada para llenar el archivo csv: vectores_terminos_de
    -pendientes.csv . La referencia acontinuacion debe ser utilizada,
    tener en cuenta que es, case-sensitive (respetar mayusculas y
    minusculas).
        * Vectores distorsionadores: S1, S2, ..., Sn
        * Vector Transpuestos de ceros: CeroV

    Nota: Para ingresar los vectores transpuestos de ceros se puede
    dejar vacio el campo o ingresar directamente respetando mayus-
    culas y minusculas.
    """
    print(load_vectores_terminos_dependientes.__doc__)
    n_dimension_matriz_gauss = conf.data['vars']['n_dimension_matriz_gauss']
    v_ter_dependientes = []
    for i in range(n_dimension_matriz_gauss):
       v_ter_dependientes.append(['Vg' + str(i+1) + '='])

    for i, fila in enumerate(v_ter_dependientes):
        aux = input(fila[0])
        if aux == '':
            aux = 'CeroV'
        v_ter_dependientes[i][0] = fila[0] + aux
        print(f"Ingreso: {v_ter_dependientes[i][0]}")

    filename = 'csv/' + conf.data['env']['path'] + '/vectores_terminos_dependientes.csv'
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(v_ter_dependientes)
    conf.data['csv_filled']['vectores_terminos_dependientes'] = True
    conf.update_config_file()
    print(f"El archivo {filename} ha sido creado con exito!")

def load_vectores_transpuestos():
  logger.debug('Ici dans le:load_vectores_transpuestos')

def run():
    # Busca los archivos csv que ya hayan sido llenado
    print(logger)
    for name in conf.data['csv_filled']:
        function = "load_" + name + "()"
        if not conf.data['csv_filled'][name]:
            logger.debug(f"Llamando a {function}")
            # Si no ha sido llenado se llama la funcion respectiva para llenarlo
            eval(function)
