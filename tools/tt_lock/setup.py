from setuptools import setup, find_packages

setup(
    name='tt_lock',
    version='0.1.0',
    author='mcrl',
    description='A simple blocking file lock library.',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)