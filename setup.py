from setuptools import setup, find_packages

__version__ = '0.1.0'

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='reactome_gnn',
    version=__version__,
    description='Python client for obtaining the pathway graph embeddings with graph neural networks.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/reactome/reactome_gnn',
    author='Lovro Vrcek',
    author_email='vrcek.lovro@gmail.com',
    license='Apache',
  
    packages=find_packages(),
    entry_points={

    },
    install_requires=[

    ],
    tests_require=['pytest'],
    classifiers=[

    ],
    platforms=['any'],
    python_requires='>=3.8',
)