from setuptools import setup, find_packages
from os import path
working_directory = path.abspath(path.dirname(__file__))

with open(path.join(working_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyimpBB',
    version='0.0.4',
    description='Implementation of an algorithm from the field of global optimization according to the publication ‘The improvement function in branch-and-bound methods’ by P. Kirst, S. Schwarze and O. Stein.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Marc Rodestock',
    author_email="marc@fam-rodestock.de",
    #url='',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent'
    ],
    packages=find_packages(),
    install_requires=['numpy','matplotlib','pyinterval','scipy'],
    #python_requires='<=3.9'
)