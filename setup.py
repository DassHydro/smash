from setuptools import setup

import versioneer

# Use setup.py to only enable versioneer else use pyproject.toml
setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
