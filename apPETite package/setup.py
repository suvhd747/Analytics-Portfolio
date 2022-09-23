import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="apPETite_UIUCiGEM_2021",
    version="0.0.1",
    author="Suva Narayan",
    author_email="vantagesuvgt8@gmail.com",
    description="Python package for the computational portion of the Illinois iGEM Project, apPETite.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UIUCiGEM/apPETite-package",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    include_package_data=True
    # package_data={"":["*.csv", "*.txt"]}
)