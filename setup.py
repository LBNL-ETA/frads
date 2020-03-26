import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="frads",
    version="0.1.1",
    author="LBNL",
    author_email="taoningwang@lbl.gov",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LBNL-ETA/frads",
    #packages=setuptools.find_packages(include=['frads']),
    packages=['frads'],
    package_dir={'frads': 'frads'},
    package_data={'frads':['data/*.*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    scripts=['bin/genmtx.py','bin/radm.py','bin/gengrid.py','bin/varays.py',
             'bin/genfmtx.py']
)
