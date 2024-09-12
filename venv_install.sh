#!/bin/bash
#bash script to install SMASH in a virtual environnement using Python venv.
#The installation has been tested for ubuntu 22.04 only
#When running this script, you will be prompted for your administrator password to install the required dependencies.

venv_path=~/python_venv
env_name=smash_dev_1.0


#Parse input arguments
i=1
until [ $# = 0 ]
do
case $1 in
    --venv_path)
    shift
    venv_path=$1
    echo "venv_path="$venv_path
    shift
    ;;
    --env_name)
    shift
    env_name=$1
    echo "env_name="$env_name
    shift
    ;;
    --help | -h)
    echo "Installation of the SMASH python package"
    echo ""
    echo "Options :" 
    echo "-h, --help                    Display the help"
    echo "--venv_path {venv_path}       Specify the folder where the python environnement will be created. Default value is $venv_path"
    echo "--env_name {env_name}         Specify the name of the python environment that will be created. Default is $env_name"
    exit 0  
    ;;
    *)
    echo "Unknown options ..."
    echo "Use --help to see available options."
    exit 0
    ;;
esac
done


if [ ! -d ${venv_path} ] ; then
    echo "Creating the folder to store the python environnement in ${venv_path}"
    mkdir ${venv_path}
fi

#install debian dependencies
#Warning to the user
if [ $(cat /etc/*release | grep ^NAME | grep -o Ubuntu) == "Ubuntu" ]; then

    DEB_PACKAGE_NAME="gfortran build-essential python3-venv"
    echo "==============================================="
    echo "Make sure you first installed the following package (ubuntu): $DEB_PACKAGE_NAME"
    echo "==============================================="
    
else

    PACKAGE_NAME="gfortran build-essential python3-venv python3-dev"
    echo "==============================================="
    echo "Builging smash require some packages. Make sure you install them first. Install the corresponding package depending your distribution. On Debian/Ubuntu these package are: $PACKAGE_NAME"
    echo "==============================================="
    
fi


echo ""
read -p "Press enter to continue, ctrl+c to abort ..."


if [ ! -d "${venv_path}/.venv-${env_name}" ] ; then
    
    echo ''
    echo 'Creating a virtual python environment for Smash...'
    echo ''
    
    #creating a python environment and activate it
    python3 -m venv "${venv_path}/.venv-${env_name}"
    ln "${venv_path}/.venv-${env_name}/bin/activate" "${venv_path}/${env_name}"
    source ${venv_path}/${env_name}
    
    #install python dependencies
    pip install --upgrade pip
    pip install -r requirements-dev.txt
    
    echo ''
    echo 'Building Smash...'
    echo ''
    
    make
    
else
    echo ''
    echo "The python environnemnt ${venv_path}/${env_name} already exist..."
    echo ''
fi
echo '************************************************'

echo ''

echo '************************************************'
echo 'To activate the python environment for Smash, run:'
echo "> source ${venv_path}/${env_name}"
echo 'To deactivate the python environnement, juste type:'
echo '>deactivate'
echo ''
echo 'To use the smash package run the following commands:'
echo "> source ${venv_path}/${env_name}"
echo '>Then launch python:'
echo '(smash)> python'
echo 'Import the SMASH package in your python shell:'
echo '>>> import smash'
echo '************************************************'