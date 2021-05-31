#!/bin/bash

resetenv=false

while getopts "r" OPT
do
    case "$OPT" in
        r)	
		resetenv=true
		;;
    esac
done

if [ ! -d ~/ska-sdc-2/data ]
then
    ln -s /project/sm47/data $HOME/ska-sdc-2/data
    mkdir $SCRATCH/processed
    ln -s $SCRATCH/processed $HOME/ska-sdc-2/processed
fi

module load daint-gpu
module load PyTorch/1.7.1-CrayGNU-20.11

if [ ! -d $HOME/env ]
then
	python -m venv --system-site-packages $HOME/env
	source $HOME/env/bin/activate
	pip install ipykernel==5.1.1 --no-deps
	pip install -r $HOME/ska-sdc-2/environment/requirements.txt
	
	# Install SoFiA
	cur_dir=$PWD
	cd $HOME
	wget https://github.com/SoFiA-Admin/SoFiA/archive/refs/tags/v1.3.2.tar.gz
	tar -xzvf v1.3.2.tar.gz
	cd SoFiA-1.3.2
	rm -rf build
	python setup.py build --force --no-gui=True
	cp -r build/lib.linux-x86_64-3.8/sofia $HOME/env/lib/python3.8/site-packages/sofia
	rm v1.3.2.tar.gz
	cd $cur_dir

	module load jupyter-utils
	kernel-create -f -n env
	cp $HOME/ska-sdc-2/environment/launcher $HOME/.local/share/jupyter/kernels/env/launcher
	cp $HOME/ska-sdc-2/environment/.jupyterhub.env .jupyterhub.env
elif [ "$resetenv" = true ]
then
	rm -r $HOME/env
	
	if [ -d $HOME/.local/lib/python3.8 ]
	then
		rm -r $HOME/.local/lib/python3.8
	fi
	
	source $HOME/ska-sdc-2/environment/daint.bash
	source $HOME/env/bin/activate
else
	source $HOME/env/bin/activate
fi