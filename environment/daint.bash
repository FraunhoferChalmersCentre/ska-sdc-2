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