from collections import UserDict
import os
import yaml
import logging as log
log.basicConfig(level=log.DEBUG, format=log.BASIC_FORMAT)

__doc__ = """
    Este modulo ha sido creado por la necesidad de tener un solo lugar
    en donde guardar los parametros tales como: path, etc.
    Para todos los demas modulos
"""


class Settings(UserDict):
    """
    Clase que maneja y actualiza el archivo de configuracion .yaml.
    """
    def __init__(self, problema_plano='default'):
          super().__init__()
          print(self.data)

conf = Settings('pedro')

def main():
    print(conf, end='\n\n')

if __name__ == '__main__':
    main()
