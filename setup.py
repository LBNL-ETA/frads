import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="frads",
    version="0.2.2",
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
        'console_scripts': ['mrad=frads.mrad:main',
                            'genmtx=frads.genmtx:main',
                            'ep2rad=frads.epjson2rad:epjson2rad_cmd',
                            'eprad=frads.eprad:main',
                            'genfmtx=frads.genfmtx:main',
                            'gengrid=frads.radutil:gengrid',
                            'genglazing=frads.genglazing:main',
                            'getwea=frads.makesky:getwea',
                            'rpxop=frads.mtxmult:rpxop',
                            'varays=frads.radutil:varays',
                            'genradroom=frads.room:genradroom',
                            'geombsdf=frads.geombsdf:main',
                            'dctsnp=frads.mtxmult:dctsnp',
                            'rglaze=frads.radutil:glaze',
                            ],
    }
)
