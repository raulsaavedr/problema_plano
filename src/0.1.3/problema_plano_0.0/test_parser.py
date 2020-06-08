import time
init_time = time.time()

import csv
filename = 'csv/ejemplo/matriz_coeficientes.csv'
coeficientes = list(csv.reader(open(filename)))

def clean(expr_mod):
    return [i for ext in [list(filter(None, termino.split(' '))) for termino in expr_mod] for i in ext ]
lista = []
for row in coeficientes:
    for expression in row:
        expr_mod = expression.split('*')
        expr_mod = clean(expr_mod)
        print(expr_mod'')
print(time.time() - init_time)