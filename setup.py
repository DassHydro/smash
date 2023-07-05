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
        "smash.solver",
        "smash.factory",
        "smash.factory.mesh",
        "smash.factory.dataset",
        "smash.factory.net",
        "smash.io",
        "smash.tests",
        "smash.tests.core",
        "smash.tests.mesh",
        "smash.tests.io",
        "smash.tests.dataset",
        "smash.tools",
    ],
    package_data={
        "smash": ["smash/solver/_solver*.so", "smash/factory/mesh/_mesh*.so"]
    },
    include_package_data=True,
    install_requires=[
        "f90wrap",
        "numpy>=1.13",
        "pandas",
        "matplotlib",
        "h5py",
        "tqdm",
        "gdal",
        "scipy",
        "pyyaml",
        "terminaltables",
    ],
    zip_safe=False,
)
