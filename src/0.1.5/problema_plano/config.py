import os
import yaml
import logging
logger = logging.getLogger('config')
logger.setLevel(logging.INFO)
logger.disabled = False

__doc__ = """
    Este modulo ha sido creado por la necesidad de tener un solo lugar
    en donde guardar los parametros tales como: path, debugging.
    Para todos los demas modulos
"""

class Settings():
    """
    Clase que maneja y actualiza el archivo de configuracion .yaml.
    se maneja internamente como un diccionario (data).
    """
    def __init__(self):
        self.data = {}
        self.default_config_file = 'conf/p_plano_default.yaml'

    def __str__(self):
        return str(self.data)

    def load(self, problema_plano='ejemplo'):
        self.config_file = f'conf/p_plano_{problema_plano}.yaml'
        # Si es la primera vez que se ejecuta cree el directorio csv
        if not os.path.exists(f'csv/{problema_plano}'):
            os.makedirs(f'csv/{problema_plano}/salida')
        # Cree el directorio graficas
        if not os.path.exists(f'graficas/{problema_plano}'):
            os.makedirs(f'graficas/{problema_plano}/flujos/surf')
            os.makedirs(f'graficas/{problema_plano}/matrices cuadradas de acoplamiento')
            os.makedirs(f'graficas/{problema_plano}/matrices diagonales de funciones hiperbolicas')
            os.makedirs(f'graficas/{problema_plano}/matrices diagonales de valores eigen')
            os.makedirs(f'graficas/{problema_plano}/potenciales/surf')
            os.makedirs(f'graficas/{problema_plano}/vectores distorsionadores')
            os.makedirs(f'graficas/{problema_plano}/vectores eigen')
        # Si el archivo de configuracion existe
        if os.path.isfile(self.config_file):
            with open(self.config_file) as f:
                # Carguelos en el diccionario
                self.data.update(yaml.full_load(f))
            logger.debug("open")
            logger.debug(self.config_file)
        # Si no existe proceda a crearlo a partir del archivo default
        else:
            logger.debug("creating")
            # Cargue el archivo default y carguelo en el diccionario data
            with open(self.default_config_file) as f:
                self.data.update(yaml.full_load(f))
            # Cambie los valores del config y del path
            self.data['env']['config_file'] = self.config_file
            self.data['env']['path'] = problema_plano
            # Escriba en el archivo las configuraciones hechas
            self.update_config_file()

    def update_config_file(self):
       with open(self.config_file, "w") as f:
             yaml.dump(self.data, f)

    def get_logging_level(self, module='main'):
        """Retorna el nivel de debug de module"""
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
            }
        level = levels[self.data['debugging'][module]['log_level']]
        return level

conf = Settings()

def main():
    logger.debug("Executing main function in config")
    conf.load(problema_plano='raul')
    conf.update_config_file()
    print(conf)

if __name__ == '__main__':
    main()
