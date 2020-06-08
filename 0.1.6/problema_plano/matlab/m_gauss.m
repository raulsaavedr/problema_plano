function [A, b, matriz] = m_gauss()
% Comparacion de solucion A\b matriz gauss - matlab
A = csvread('../csv/ejemplo/salida/matriz_coeficientes.csv');
b = csvread('../csv//ejemplo/vectores_valores_dependientes.csv');
matriz = A\b;
