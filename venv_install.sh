#!/bin/bash
#bash script to install SMASH in a virtual environnement using Python venv.
#The installation has been tested for ubuntu 22.04 only
#When running this script, you will be prompted for your administrator password to install the required dependencies.

venv_path=~/python_venv/
env_name=smash

if [ ! -d ${venv_path} ] ; then
    mkdir ${venv_path}
fi

#install debian dependencies
#we should support other distribution here... and check the os version

if cat /etc/*release | grep ^NAME | grep Ubuntu; then
    DEB_PACKAGE_NAME="gfortran build-essential gdal-bin libgdal-dev libshp-dev python3-gdal python3-venv python3-dev"
    echo "==============================================="
    echo "Installing packages $DEB_PACKAGE_NAME on Ubuntu"
    echo "==============================================="
    sudo apt install $DEB_PACKAGE_NAME
else
    echo "unsupported os. Please install manually the corresponding package (Ubuntu/Debian) for your distribution."
    echo "sudo apt install $DEB_PACKAGE_NAME"
fi

if [ ! -d ${venv_path}/${env_name} ] ; then
    
    echo ''
    echo 'Creating a virtual python environment for Smash...'
    echo ''
    
    #creating a python environment and activate it
    python3 -m venv "${venv_path}/.venv-${env_name}"
    ln "${venv_path}/.venv-smash/bin/activate" "${venv_path}/${env_name}"
    source ${venv_path}/${env_name}
    
    #install minimal python dependencies
    #here we should list and install all dependencies ?
    pip install --upgrade pip
    pip install 'numpy>=1.13,<=1.23'
    pip install f90wrap
    pip install wheel
    
    #manually intalling gdal, because it depends on the version of the installed system library
    pip install GDAL<=$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
    
    echo ''
    echo 'Building Smash...'
    echo ''
    #building the code inside the python environnement. This will automatically install all others python dependencies
    make
    
    echo ''
    echo 'Installing extra-package for building the documentation...'
    echo ''
    #install extra-package for building the documentation:
    pip install sphinx numpydoc pydata-sphinx-theme ipython sphinxcontrib-bibtex sphinx-design sphinx-autosummary-accessors pytest black
    
else
    echo ''
    echo "The python environnemnt smash already exist. Remove the directory ${venv_path}/${env_name} if you need to  re-install Smash."
    echo ''
fi
echo '************************************************'

echo ''

echo '************************************************'
echo 'To activate the python environment for Smash run:'
echo "> source ${venv_path}/${env_name}"
echo 'To deactivate the python environnement, juste type:'
echo '>deactivate'
echo 'To use the smash package run the following commands:'
echo "> source ${venv_path}/${env_name}"
echo '(smash)> python'
echo '>>> import smash'
echo '************************************************'

