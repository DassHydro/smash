from setuptools import setup

setup(
    name="smash",
    version="0.1.0",
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
    install_requires=["numpy>=1.21.2", "pandas>=1.4.2", "h5py>=3.7.0", "tqdm>=4.64.0"],
    zip_safe=False,
)
