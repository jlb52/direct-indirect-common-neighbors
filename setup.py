from setuptools import setup, find_packages

setup(
    name='dicn',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'networkx',
        'numpy',
    ],
    author='Jackson Burke',
    description="An implementation of the Direct-Indirect Common Neighbors algorithm",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/jlb52/direct-indirect-common-neighbors"
)