import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="frads",
    version="0.2",
    author="LBNL",
    author_email="taoningwang@lbl.gov",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LBNL-ETA/frads",
    packages=setuptools.find_packages(include=['frads', 'frads.sundry']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
