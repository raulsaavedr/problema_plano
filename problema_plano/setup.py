from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="problema_plano", # Replace with your own username
    version="1.0.0",
    author="Raul Saavedra, Daniel Useche",
    author_email="raulsaavedr@gmail.com, danieluseche@gmail.com",
    description="Calculo del problema plano",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raulsaavedr/problema_plano",
    packages=find_packages(),
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
