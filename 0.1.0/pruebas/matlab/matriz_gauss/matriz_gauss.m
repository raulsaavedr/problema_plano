function [mat_gauss] = matriz_gauss()
    % Prueba de matriz de 5x5
    mat_acoplamiento = [1 2 3 0 0;
                        0 3 4 5 0;
                        0 0 5 6 7;
                        0 0 0 7 8;
                        0 0 1 0 4;];
    vector_terminos_dependientes = [1;
                                    0;
                                    5;
                                    0;
                                    1;];
    mat_gauss = mat_acoplamiento \ vector_terminos_dependientes ;