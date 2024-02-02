from setuptools import setup, find_packages

setup(
    name="BRESTMultiLabelOpthalmology",
    version="1.0.0",
    description="Multi-label classification of retinal fundus images.",
    author="Dewi Gould",
    author_email="dewi.gould@maths.ox.ac.uk",
    url = "",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "numpy",
        "pandas",
        "sklearn",
        "skimage",
        "pickle",
        "os",
        "random",
        "copy",
        #CONTINUE --- TODO
    ],
)
