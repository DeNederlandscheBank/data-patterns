#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['pandas', 'numpy', 'xlsxwriter']

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="De Nederlandsche Bank",
    author_email='ECDB_berichten@dnb.nl',
    python_requires='>=3.0, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Package for generating and evaluating patterns in quantitative reports",
    install_requires=requirements,
    license="MIT/X license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='data_patterns',
    name='data_patterns',
    packages=find_packages(include=['data_patterns', 'data_patterns.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/DeNederlandscheBank/data-patterns',
    version='0.1.6',
    zip_safe=False,
)
