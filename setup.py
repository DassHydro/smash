from setuptools import setup
import versioneer

setup(
    name="smash",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Spatially distributed Modelling and ASsimilation for Hydrology",
    url="/",
    author="INRAE",
    packages=[
        "smash",
        "smash.core",
        "smash.core.optimize",
        "smash.solver",
        "smash.mesh",
        "smash.io",
        "smash.dataset",
        "smash.tests",
        "smash.tests.core",
    ],
    package_data={"smash": ["smash/solver/_solver*.so", "smash/mesh/_meshing*.so"]},
    include_package_data=True,
    install_requires=[
        "f90wrap",
        "numpy<=1.23", # pinned due to f90wrap 0.2.9 not handling numpy 1.24
        "pandas",
        "matplotlib",
        "h5py",
        "tqdm",
        "gdal",
        "scipy",
        "pyyaml",
        "SALib",
        "terminaltables",
    ],
    zip_safe=False,
)
