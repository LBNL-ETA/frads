import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="frads",
    version="0.2.5",
    author="LBNL",
    author_email="taoningwang@lbl.gov",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LBNL-ETA/frads",
    packages=['frads'],
    package_dir={'frads': 'frads'},
    package_data={'frads': ['data/*.*', "data/standards/*.*"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': ['mrad=frads.cli:mrad',
                            # 'genmtx=frads.cli:main',
                            'ep2rad=frads.cli:epjson2rad_cmd',
                            # 'eprad=frads.eprad:main',
                            # 'genfmtx=frads.genfmtx:main',
                            'gengrid=frads.cli:gengrid',
                            'rpxop=frads.cli:rpxop',
                            'varays=frads.cli:varays',
                            'genradroom=frads.cli:genradroom',
                            'geombsdf=frads.geombsdf:main',
                            'dctsnp=frads.cli:dctsnp',
                            'glazing=frads.cli:glazing',
                            'gencolorsky=frads.gencolorsky:main',
                            ],
    }
)
