import setuptools

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
    package_data={'frads': ['data/*.*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': ['mrad=frads.mrad:main',
                            'genmtx=frads.genmtx:main',
                            'ep2rad=frads.epjson2rad:main',
                            'eprad=frads.eprad:main',
                            'genfmtx=frads.genfmtx:main',
                            'gengrid=frads.radutil:gengrid',
                            'getwea=frads.makesky:getwea',
                            'rpxop=frads.radutil:rpxop',
                            'varays=frads.radutil:varays',
                            'genradroom=frads.room:genradroom',
                            'geombsdf=frads.geombsdf:main',
                            ],
    }
)
