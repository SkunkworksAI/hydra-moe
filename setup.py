# -*- coding: utf8 -*-
#
# Skunkworks AI 2023
#
#

import os

from setuptools import setup, find_packages

# Meta information
version = open('VERSION').read().strip()
dirname = os.path.dirname(__file__)

# Save version and author to __meta__.py
data = '''# Automatically created. Please do not edit.
__version__ = u'%s'
__author__ = u'F\\xe1bio Mac\\xeado Mendes'
''' % version
with open(path, 'wb') as F:
    F.write(data.encode())
    
setup(
    # Basic info
    name='python-boilerplate',
    version=version,
    author='SkunkworksAI Authors', 
    url='https://github.com/SkunkworksAi/hydra-moe',
    description='Open source AI.',
    long_description=open('README.rst').read(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries',
    ],

    # Packages and depencies
    package_dir={'': 'src'},
    packages=find_packages('src'),
    install_requires=[
        'jinja2',
        'invoke>=0.13',
        'unidecode',
        'six',
    ],
    extras_require={
        'dev': [
            'manuel',
            'pytest',
            'pytest-cov',
            'coverage',
            'mock',
        ],
    },

    # Data files
    package_data={
        'python_boilerplate': [
            'templates/*.*',
            'templates/license/*.*',
            'templates/docs/*.*',
            'templates/package/*.*'
        ],
    },

    # Scripts
    entry_points={
        'console_scripts': [
            'python-boilerplate = python_boilerplate.__main__:main'],
    },

    # Other configurations
    zip_safe=False,
    platforms='any',
)