function [A, b, matriz] = m_gauss()
% Comparacion de solucion A\b matriz gauss - matlab
A = csvread('../csv/salida/matriz_coeficientes_np.csv');
b = csvread('../csv/salida/vectores_valores_dependientes_np.csv');
matriz = A\b;