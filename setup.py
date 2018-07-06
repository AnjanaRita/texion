from io import open

from setuptools import find_packages, setup

with open('texion/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split('=')[1].strip(' \'"')
            break
    else:
        version = '0.0.1'

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

REQUIRES = []

setup(
    name='texion',
    version=version,
    description='pytorch and scikit-learn based text classification library',
    long_description=readme,
    author='Bharath G.S',
    author_email='royalkingpin@gmail.com',
    maintainer='Bharath G.S',
    maintainer_email='royalkingpin@gmail.com',
    url='https://github.com/bharathgs/texion',
    license='MIT',

    keywords=[
        'pytorch', 'text', 'text classification', 'NLP', 'scikit-learn',
        'sklearn', 'machine learning', 'deep learning'
    ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],

    install_requires=REQUIRES,
    tests_require=['coverage', 'pytest'],

    packages=find_packages(),
)
