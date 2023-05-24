#!/usr/bin/bash

python -m venv venv
source venv/bin/activate

# Python packages
pip3 install numpy pandas matplotlib random sklearn scipy json datetime

# PRISM data
wget https://ndownloader.figshare.com/files/20237709

# PRISM readme
#wget https://ndownloader.figshare.com/files/20237700
