#!/bin/bash
# bash script to install smash in a virtual environnement using Python venv.

env_name=smash-dev

if [ ! -d ${venv_path}/${env_name} ] ; then
    
    echo ''
    echo "Creating a virtual environment for smash..."
    echo ''
    
    # Creating a python environment and activate it
    python3 -m venv "${env_name}"
    . ${env_name}/bin/activate
    
    echo ''
    echo "Building smash..."
    echo ''
    # Building the code inside the python environnement.
    pip install -r requirements-dev.txt
    make
    
else
    echo ''
    echo "The environnemnt ${env_name} already exists. Remove the directory ${env_name} if you need to re-install smash."
    echo ''
fi
echo '************************************************'

echo ''

echo '************************************************'
echo "To activate the environment:"
echo "> . ${env_name}/bin/activate"
echo "To deactivate the environnement:"
echo "> deactivate"
echo "To use the smash package:"
echo "> . ${env_name}/bin/activate"
echo "(${env_name})> python"
echo ">>> import smash"
echo '************************************************'

