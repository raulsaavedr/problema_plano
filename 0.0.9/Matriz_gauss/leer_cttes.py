import csv
constantes = []
with open('csv/constantes_c.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        constantes.append(row)
maxima_const = 0
for list_constante in constantes:
    for constante in list_constante:
        if int(constante.split('C')[1]) > maxima_const:
            maxima_const = int(constante.split('C')[1])
constantes_int = [[0 for i in range(1, maxima_const + 1)] for j in range(1, maxima_const + 1)]
# Se crea una lista de listas con unos donde estan C1,C2,CN y ceros donde no hay valores
j = 0
for list_constante in constantes:
    for constante in list_constante:
        constantes_int[j][int(constante.split('C')[1]) - 1] = 1
    print(constantes_int[j])
    j += 1

#print(np.array(constantes_int))
