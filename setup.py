from setuptools import setup, find_packages
from os import path
working_directory = path.abspath(path.dirname(__file__))

with open(path.join(working_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyimpBB',
    version='0.0.6',
    description='Implementation of an algorithm from the field of global optimization according to the publication ‘The improvement function in branch-and-bound methods’ by P. Kirst, S. Schwarze and O. Stein.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Marc Rodestock',
    author_email="marc@fam-rodestock.de",
    #url='https://github.com/uisab/pyimpBB',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent'
    ],
    keywords='global optimization, improvement function, branch-and-bound method',
    packages=find_packages(),
    #python_requires='==3.9',
    install_requires=['numpy','matplotlib','pyinterval','scipy'],
    project_urls={
    #    "The Publication": "",
        "Institution": "https://www.ior.kit.edu",
        "Source": "https://github.com/uisab/pyimpBB"
    }
)