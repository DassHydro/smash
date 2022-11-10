from setuptools import setup

setup(
    name="smash",
    version="0.1.8",
    description="Spatially distributed Modelling and ASsimilation for Hydrology",
    url="/",
    author="Francois Colleoni",
    packages=[
        "smash",
        "smash.core",
        "smash.solver",
        "smash.mesh",
        "smash.io",
        "smash.dataset",
    ],
    package_data={"smash": ["smash/solver/_solver*.so", "smash/mesh/_meshing*.so"]},
    include_package_data=True,
    install_requires=[
        "f90wrap",
        "numpy",
        "pandas",
        "matplotlib",
        "h5py",
        "tqdm",
        "gdal",
        "scipy",
        "pyyaml",
    ],
    zip_safe=False,
)
