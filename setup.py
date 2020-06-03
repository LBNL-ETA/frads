import setuptools
import os

bin_path = [os.path.join('bin', p) for p in os.listdir('bin') if p.endswith('.py')]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="frads",
    version="0.2.1",
    author="LBNL",
    author_email="taoningwang@lbl.gov",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LBNL-ETA/frads",
    packages=['frads'],
    package_dir={'frads': 'frads'},
    package_data={'frads':['data/*.*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    scripts=bin_path,
)
