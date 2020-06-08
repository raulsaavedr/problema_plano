Changelog:
VERSION 0.1.6:
  * Se agregaron las graficas de continuidad de flujos
  * Se agregaron las graficas de potenciales totales
  * Se agregaron las graficas 3D
  * Se agregaron las graficas 2D
  * Se modifico la expresion regular de pattern_eig en el modulo parser de la matriz de gauss
  * Se cambio el nombre del main a ProblemaPlano
  * Se agrego la clase Problema Plano y su metodo run
VERSION 0.1.5:
  * Se cambio completamente el modulo parser de la matriz de gauss
  * Se agrego jit de numba a la funciones integrando de los modulos distorsiona-
    dor y matriz_coeficientes
  * Se cambio la forma de crear las graficas en el modulo de graficacion, mejo-
    rando la velocidad de creacion.
  * Se eliminaron las funciones main restantes de cada modulo.
VERSION 0.1.4:
  * Se elimino la funcion main() de varios modulos.
VERSION 0.1.3:
  * Se inserto el modulo config y fill_the_csv ademas se creo el archivo de con-
    figuracion p_plano_default.yaml, se separaron los paths para poder ejecutar
    diferentes problemas plano con el mismo codigo.
  * Se agrego __init__.py y setup.py al paquete
  * Se corrigio el nombre de acomplamiento a acoplamiento
VERSION 0.1.2:
  * Se logro solucionar el error de la matriz de gauss.
VERSION  0.1.1:
  * cargar_mesh_regiones se elimino parametro de entrada: n_dimension y se agrego
    otro parametro de retorno: dimension_mesh
  * Se modifico potencia y flujos para que no dependieran de n_dimension, si no
    de la dimension del meshgrid(delta): dimension_mesh.
  * calcular_flujos y calcular_potenciales recibe ahora parametro: dimension_mesh
  * prueba_flujo y prueba_potencial recibe ahora parametro: dimension_mesh
VERSION  0.1.0:
  * Se modifico regiones.py: se modifico la funcion cargar_mesh_regiones y se
    agrego un recorte a los vectores aplanados del mesh en (x) y (y) para que
    fueran hasta n_dimension.
  * Se agrego el archivo flujo.py bajo la misma logica que el archivo potencial.py
  * Se modifico de version 1.0.0 a 0.1.0
VERSION  0.0.9:
  * Se agrego cargar_ejes_relativos y cargar_mesh_regiones al archivo regiones.py
  * Se modifico el archivo v_transpuestos ahora es mas simple
  * Se modifico el archivo potencial corrigiendo el error del calculo
  * Se modifico archivo de entrada regiones.csv ahora el campo 'Eps' = 'eps'
VERSION  0.0.8:
  * Se modifico el archivo csv region para que incluya direcciones de flujo y de
    esta forma se puedan calcular los ejes relativos de cada region. Se agrego
    cargar_ejes_relativos() al modulo region.py
  * Se corrigio el archivo v_transpuestos.py pues este no se acoplaba al calculo
    de los potenciales.
VERSION: 0.0.7
  * Se agrego el archivo potencial y vectores_transpuestos.
VERSION: 0.0.6
  * Se modifico el parser de la matriz de gauss, y se agrego funcion para
    los vectores de terminos dependientes
VERSION: 0.0.5
  * Se cambio las funciones y variables de camelCase a snake_case para que sea mas pythonic
VERSION: 0.0.4
  * Se cambio la forma de calcular analitica del modulo matrices_acoplamiento.py
    la forma anterior presentaba problemas cuando abs(Veig1-Veig2) = 0
VERSION: 0.0.3
  * Se elimino el archivo trigonometricas.py , no es necesario calcular los
    vectores traspuestos de funciones trigonometricas.
VERSION: 0.0.2
  * Se cambió los indices de los vectores eigen de Pn,Qn,...,Zn a P,Q,...,Z
VERSION: 0.0.1
  * Se cambió la forma de graficar las matrices, ya no es necesario cargar en
    memoria los DataFrame MultiIndex, si se necesita luego, ya la funcion de
    calcular matrices esta hecha tanto para valores eigen como funciones hiperbolicas