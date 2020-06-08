import setuptools

setuptools.setup(
    name="problema_plano", # Replace with your own username
    version="0.1.3",
    author="Raul Saavedra, Daniel Useche",
    author_email="raulsaavedr@gmail.com, danieluseche@gmail.com",
    description="Calculo del problema plano",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy", "pandas", "scipy", "numba",
    ]
)
