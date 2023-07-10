from setuptools import find_packages
from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()
    
setup(
    name='keras-gsam',
    version='1.0.0',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Marc Blanco',
    author_email='marcblanco03@gmail.com',
    url='https://github.com/mortfer/keras-gsam',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.10'
    ],
    python_requires='>=3',
    keywords='tensorflow keras optimization gsam loss landscape',
)
