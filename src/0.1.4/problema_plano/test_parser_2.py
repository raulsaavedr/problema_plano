import re
import csv
filename = 'csv/ejemplo/vectores_terminos_dependientes.csv'
vectores_reader = list(csv.reader(open(filename)))
vectores_valores_dependientes_list = []
for expression in vectores_reader:
    expr_mod = expression[0].split('=')[1].replace(' ', '')
    pattern = re.compile(r'S[0-9]+')
    replace = 'vectores_distorsionadores.loc[\'\g<0>\'].to_numpy()'
    expr_mod = re.sub(pattern, replace, expr_mod)
    vectores_valores_dependientes_list.append(expr_mod)

print(vectores_valores_dependientes_list)

    