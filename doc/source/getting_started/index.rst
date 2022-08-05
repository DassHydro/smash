.. _getting_started:

===============
Getting started
===============

------------
Installation
------------

`smash` can be used under Linux and should work with most distributions. It has been tested for use on Ubuntu and CentOS. The installation instructions are detailed for Ubuntu. Depending on your distribution, you will need to use the correct package manager and insert the appropriate packages.

*******
Sources
*******

Install `smash` from this gitlab repository,

.. code-block:: none

	git clone https://gitlab.irstea.fr/francois.colleoni/smash.git
	
A ``smash`` directory should've been cloned into your current working directory.

*******
Compile
*******

.. dropdown:: On your own environment
	:animate: fade-in-slide-down
	:container: + shadow
	:title: font-weight-bolder
	:open:
	
	.. image:: ../_static/logo_terminal.svg
		:width: 75
		:align: center
	
	1. **Install f90wrap**
	
		Install the `f90wrap <https://github.com/jameskermode/f90wrap>`__ package using ``pip3``
	
		.. code-block:: none
			
			sudo pip3 install f90wrap
			
		.. note ::
			
			``pip3`` can be installed as follows:
			
			.. code-block:: none
			
				sudo apt install python3-pip
				
	2. **Install gdal**
	
		Install the `gdal <https://gdal.org/api/python.html>`__ package
	
		.. code-block:: none
			
			sudo add-apt-repository ppa:ubuntugis/ppa
			sudo apt-get update
			sudo apt-get install gdal-bin libgdal-dev
			
	3. **Build**
	
		Use the `make <https://www.gnu.org/software/make/manual/make.html>`__ command inside the `smash` directory
		
		.. code-block:: none
		
			make
			
	4. **Check your installation**
		
		Once the compilation is finished, make sure that everything went well by tring to import the `smash` package inside a Python interface
		
		.. code-block:: none
		
			python3
			
		.. code-block:: python
			
			>>> import smash
	
.. dropdown:: Anaconda
	:animate: fade-in-slide-down
	:container: + shadow
	:title: font-weight-bolder
	:open:
	
	.. image:: ../_static/logo_anaconda.png
		:width: 175
		:align: center
	
	.. note::
	
		``conda`` can be downloaded `here <https://www.anaconda.com/>`__
			
	1. **Create environment**
	
		In the `smash` directory create the ``conda`` environment named ``smash-dev``:
		
		.. code-block:: none
		
			conda env create -f environment.yml
			
	2. **Activate environment**
	
		Active the environment:
		
		.. code-block:: none
		
			conda activate smash-dev
			
	3. **Build**
	
		Use the `make <https://www.gnu.org/software/make/manual/make.html>`__ command inside the `smash` directory
		
		.. code-block:: none
		
			(smash-dev) make
			
	4. **Check your installation**
		
		Once the compilation is finished, make sure that everything went well by tring to import the `smash` package inside a Python interface
		
		.. code-block:: none
		
			(smash-dev) python3
			
		.. code-block:: python
			
			>>> import smash
			
	.. hint::
		
		If you'd prefer that conda's base environment not be activated on startup, 
   		set the auto_activate_base parameter to false: 
		
		.. code-block::
		
			conda config --set auto_activate_base false
	
.. dropdown:: Docker
	:animate: fade-in-slide-down
	:container: + shadow
	:title: font-weight-bolder
	:open:
	
	.. image:: ../_static/logo_docker.png
		:width: 175
		:align: center
		
	.. warning::
		
		Section in development
