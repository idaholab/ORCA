import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ORCA",
    author="Daniel Garrett",
    author_email="daniel.garrett@inl.gov",
    description="Optimization of Real-time Capacity Allocation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=["tests*", "notebooks*"]),
    include_package_data=True,
    install_requires=["numpy", "pandas", "pyomo", "pyyaml"],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
)
