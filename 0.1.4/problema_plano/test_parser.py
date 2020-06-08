import re
import csv
filename = 'csv/raul/matriz_coeficientes.csv'
n_dimension= 100
matriz_reader = list(csv.reader(open(filename)))
# Se crean los pratones y reemplazos para cada variable
# Patrones y reemplazos de los materiales (Eps)
pattern_e0 = re.compile(r'E0')
replace_e0 = '1'
pattern_er = re.compile(r'(Er)([\* | \s])')
replace_er = '2\g<2>'
pattern_ern = re.compile(r'(Er)([1-9])')
replace_ern = str('(\g<2>+1)')
# Matriz diagonal de unos
pattern_uno = re.compile(r'Uno')
replace_uno = f"np.eye({n_dimension})"
# Matriz cuadrada de ceros
pattern_cero = re.compile(r'Cero')
replace_cero = f"np.diag(np.zeros({n_dimension}))"
# Constanstes k
pattern_k = re.compile(r'k\d+')
replace_k = "constantes_k.loc['\g<0>', 'calcular']"
# Vectores eigen
pattern_eig = re.compile(r'([A-Z])([\* | \s])')
replace_eig = "matrices_diagonales_valores_eig.loc['\g<1>'].to_numpy()\g<2>"
# Matrices diagonales de funciones hiperbolicas
pattern_hyp = re.compile(r'D\d+')
replace_hyp = "matrices_diagonales_hiperbolicas.loc['\g<0>'].to_numpy()"
# Matrices acopladoras con un digito
pattern_acop_1_dig = re.compile(r'(?<!\')(M\d)(?!(T|\'))')
replace_acop_1_dig = "matrices_acoplamiento.loc['\g<1>'].to_numpy()"
# Matrices acopladoras con dos digitos
pattern_acop_2_dig = re.compile(r'(M\d\d)(?!(T|\'))')
replace_acop_2_dig = "matrices_acoplamiento.loc['\g<1>'].to_numpy()"
# Matrices acopladoras transpuestas
pattern_acop_t = re.compile(r'(?<!\')(M\d+)(T|\')')
replace_acop_t = "matrices_acoplamiento_trans.loc['\g<1>'].to_numpy()"

# Lista que contiene cada patron y reemplazo en forma de tupla
repl_patter_list = [
    (pattern_e0, replace_e0), (pattern_er, replace_er), (pattern_e0, replace_e0),
    (pattern_ern, replace_ern), (pattern_uno, replace_uno), (pattern_cero, replace_cero),
    (pattern_k, replace_k), (pattern_eig, replace_eig), (pattern_hyp, replace_hyp),
    (pattern_acop_2_dig, replace_acop_2_dig), (pattern_acop_t, replace_acop_t),
    (pattern_acop_1_dig, replace_acop_1_dig),
    ]
    
row_list = []
matriz_coeficientes_list = []
for row in matriz_reader:
    for expression in row:
        expr_mod = expression.split('=')[1]
        for pattern, replace in repl_patter_list:
       # expr_mod = re.sub(pattern_acop_2_dig, replace_acop_2_dig, expr_mod)
        #expr_mod = re.sub(pattern_acop_1_dig, replace_acop_1_dig, expr_mod)
            expr_mod = re.sub(pattern, replace, expr_mod)
           
        row_list.append(expr_mod)
        print(expr_mod)
    matriz_coeficientes_list.append(row_list)

#print(vectores_valores_dependientes_list)

    