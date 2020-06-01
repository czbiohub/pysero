# Jetson Nano Full Setup for open-source ELISA 
## "Nautilus"-- a SQUID derivative

-----------------------------
## Overview:
1. Write ISO file and boot up Jetson Nano
2. Create a swap file
3. Upgrade software
4. Install pip3
5. Create and activate virtual environment for imaging
6. Clone octopi-platereader repository
7. Install cuttlefish dependencies
8. Install Galaxy Camera SDKs
 
-----------------------------
### 1. [Write ISO File and boot](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro "WriteIsoFile"):
Follow the instructions in the link above to write the ISO file and boot the Jetson Nano.

### 2. [Create swap file](https://medium.com/@heldenkombinat/getting-started-with-the-jetson-nano-37af65a07aab "SwapFile"): 
Create a 4G swapfile to avoid the Jetson Nano becoming unresponsive when the memory is filled up. 
1. `sudo fallocate -l 4G /var/swapfile`
2. `sudo chmod 600 /var/swapfile`
3. `sudo mkswap /var/swapfile`
4. `sudo swapon /var/swapfile`
5. `sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0" >> /etc/fstab'`
6. Check to see if it worked: `free -h`

### 3. Upgrade software
Upgrade the Jetson Nano software before installing dependencies. 
1. 	Update to download latest packages: 
	`sudo apt-get update`
2. 	Upgrade apt-get (the program that downloads software packages) with the latest packages:
	`sudo apt-get upgrade`
3. 	Reboot the system to complete the upgrade:
	`sudo reboot`

### 4. Install pip3
1. `sudo apt-get install python3-pip`
2. (Optional) [update pip](https://pip.pypa.io/en/stable/installing/#upgrading-pip "Upgrade pip"):
	`pip3 install -U pip`

### 5. Create and activate virtual environment
We prefer to install two virtual environments, one to control *Nautilus* and one to control the *Pysero* pipeline. 
First, let us install the *Cuttlefish* environment.

Following installation was tested on `4.9.140-tegra aarch64 GNU/Linux`.

We use a python package manager to isolate the dependencies required for nautilus.
conda package manager is not directly supported by above version of Linux.
Instead, pip virtual environment works well for package management.

Major steps in setup are:
1.	Install virtual environment: 
    `sudo apt install -y python3-venv`
    
2.	Create an environment: 
    `python3 -m venv ~/python-envs/nautilus`
    
3.	Activate environment: 
    `source ~/python-envs/nautilus/bin/activate`
    
4.	Deactivate environment: 
    `deactivate`
    
5. Update pyenv.cfg to use system-wide packages:
    `cd ~/python-envs/nautilus`
    
    Use `nano pyvenv.cfg to` change:
    `include-system-site-packages = true` 
    (allow packages that require lower-level installation to be accessed in the environment)

Some packages need to be installed in local environment and some system-wide.
Use `pip install -I` to install in the `pysero` site-packages folder, the environment will access the local pacakages first and then search for global packages.

### Install dependencies for python packages:

First we install the dependencies for _main.py_ 
1. `sudo -H pip3 install -I qtpy`
2. `sudo -H pip3 install -I pyqtgraph`
3. `sudo -H pip3 install -I pyserial`

Note: you may need to install over an existing PyYAML file. Use the flag `--ignore-installed PyYAML`

Example:
- `sudo -H pip3 install -I serial --ignore-installed PyYAML`
or
- `sudo -H pip3 install -I pyserial --ignore-installed PyYAML`

## [Install camera software](https://www.get-cameras.com/customerdownloads "Camera Software"):
Next we install camera software from the link above. 
Download both the *Linux ARM SDK* and the *Linux Python SDK*. Extract the packages and follow the readmes (written below). 

### Install the ARM SDK first: 

1. Installation command: `./Galaxy_camera.run`

2. When you do not see any error message in the installation and the end shown as below, your installation is compelete.


All configurations will take effect after the system is rebooted.
If you don't want to reboot the system for a while,
you will need to unplug and replug the camera before using SDK.


3. If you would like to run GxViewer or GxGigeIPConfig, You have to make sure that Qt4 has been installed in your system. 
    Install Qt4 command in some system: sudo apt-get install qt4-default  

Notes:
    1. After your installation, You need to replug Galaxy USB3 device before using SDK.
    2. Don't install the SDK in a folder which path has chinese characters, or SDK may not work. 



------------------------------
### Then install the Python SDK:
Python3.5 gxipy installation  
====================

1. Install python3.5 & python3.5-dev

  (1) `sudo apt-get install python3.5`
  
  (2) `sudo apt-get install python3.5-dev`

2. Install the python3-setuptools toolkit
 
  (1) `sudo apt-get install python3-setuptools`
  
3. Install gxipy library

  (1) `cd ./api`
  
  (2) `python3 setup.py build`
  
  (3) `sudo python3 setup.py install`

4. Install numpy library  
     1) `sudo apt-get install python3-pip`
     2) `sudo pip3 install cython`
     3) `sudo pip3 install numpy`


-------------------------
# Pysero installation: 

## Linux/Mac/Windows (x64)
1. Create a new conda environment.
2. Clone the repository.
3. `conda install --file requirements. txt`

## Jetson Nano (aarch64)

Following installation was tested on `4.9.140-tegra aarch64 GNU/Linux`.

We use a python package manager to isolate the dependencies required for pysero.
conda package manager is not directly supported by above version of Linux.
Instead, pip virtual environment works well for package management.

Major steps in setup are:
1.	Install virtual environment: 
    `sudo apt install -y python3-venv`
    
2.	Create an environment: 
    `python3 -m venv ~/python-envs/pysero`
    
3.	Activate environment: 
    `source ~/python-envs/pysero/bin/activate`
    
4.	Deactivate environment: 
    `deactivate`
    
5. Update pyenv.cfg to use system-wide packages:
    `cd ~/python-envs/pysero`
    
    Use `nano pyvenv.cfg to` change:
    `include-system-site-packages = true` 
    (allow packages that require lower-level installation to be accessed in the environment)

Some packages need to be installed in local environment and some system-wide.
Use `pip install -I` to install in the `pysero` site-packages folder, the environment will access the local pacakages first and then search for global packages.

### Install dependencies for python packages:

scikit-image and scipy dependencies:

        sudo apt-get install python-dev libfreetype6-dev
	sudo apt-get install libfontconfig1-dev libjpeg-dev
	sudo apt-get install zlib1g-dev
	sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran

	
### Install python packages:
	
	
	pip3 install -I wheel
	pip3 install -I pillow
	pip install -I scipy==1.1.0
	# compilation of scipy can take >5 min.
	pip install -I scikit-image 
	# compilation of scikit-image takes >5min. May have to try sudo -H pip3 install scikit-image
	pip install -I matplotlib 
	pip install -I pandas==0.24.0 
	# compilation of pandas can take time.
	pip install -I certifi
	pip install -I cycler
	pip install -I kiwisolver
	pip install -I cython
	pip install -I numpy==1.18.1
	pip install -I python-dateutil
	pip install -I pytz
	pip install -I six
	pip install -I tabulate
	pip install -I tornado
	pip install -I openpyxl
	pip install -I xmltodict
	pip install -I xlrd

### Note: 
I recommend installing these packages slowly and one by one. Sometimes it may be useful to use `sudo -H pip3 install...`. I also recommend doing them in this order.
