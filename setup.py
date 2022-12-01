from setuptools import setup
import versioneer

setup(
    name="smash",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Spatially distributed Modelling and ASsimilation for Hydrology",
    url="/",
    author="Francois Colleoni",
    packages=[
        "smash",
        "smash.core",
        "smash.core.optimize",
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
