from setuptools import setup, find_packages

setup(
    name="problema_plano", # Replace with your own username
    version="1.0.0",
    author="Raul Saavedra, Daniel Useche",
    author_email="raulsaavedr@gmail.com, danieluseche@gmail.com",
    description="Calculo del problema plano",
    long_description="El problema plano es un problema de valores de frontera,\
                      se buscar analizar el comportamiento de los potenciales y\
                      flujos electricos de una linea o ducto de transmision que\
                      esta subdivida por multiples regiones de diferentes materiales\
                      dielectricos, Se muestra la interaccion entre cada region tanto\
                      de los flujos como de los potenciales.",
    long_description_content_type="text/markdown",
    url="https://github.com/raulsaavedr/problema_plano",
    packages=setuptools.find_packages(),
   classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Data Scientist:: Students',
        'Topic :: Data Sciencie :: Computing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        "Operating System :: OS Independent",
    ],    
    python_requires='>=3.7',
    install_requires=[
        "numpy", "pandas", "scipy", "numba",
    ]
)
