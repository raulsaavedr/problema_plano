import numpy as np

def answer():
	n = 5 
	xoriginal = np.array(\
				[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
				 [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
				 [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
				 [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 
				 [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0], 
				 [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0], 
				 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 
				 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], 
				 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], 
				 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
				 [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
				 [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
				 [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 
				 [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0], 
				 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1]])
	# Vector columna que contiene la suma de los elementos de cada
	# fila ordenado de menor a mayor
	index_sorted= np.argsort(xoriginal.sum(axis=1))
	x = xoriginal[index_sorted]
	# Y ahora encuentre la solucion
	per = first(x, -1, np.arange(0, n))
	print(f"\t\nSolucion inicial\n{xoriginalx}")
	print(f"\t\nShuffled version\n{x}")
	if not np.isnan(per.any()):
		print(f"\t\nSolucion encontrada\n{x[per,:]}")
		print(f"\t\nOreden de las filas\n{per}")
	else:
		print(f"\t\nSolucion no encontrada!\n")
		return

def first(x, row, whichrowstoplace):
	solution = np.empty(0)
	for i in range(0, len(x) - row - 1):
		solution = next(x, row + 1, i, whichrowstoplace)
		if not np.isnan(solution): # OJO AQUI
			return
	return solution

def next(x, row , i, whichrowstoplace):
	firstrow = x[row,:]
	onelocation =  np.nonzero(firstrow)[0] # OJO AQUI
	leaves = onelocation[np.isin(onelocation, whichrowstoplace)]
	if i > len(leaves):
		solution = np.nan
	else:
		solution = np.append(solution, first(x, row, np.delete(whichrowstoplace, leaves)))

	return solution


def main():
	answer()


if __name__ == '__main__':
	main()