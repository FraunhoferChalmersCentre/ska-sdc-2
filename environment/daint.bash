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

if [ ! -d ~/project ]
then
	ln -s /project/sm47 project
fi

module load daint-gpu
module load PyTorch/1.6.0

if [ ! -d ~/env ]
then
	python -m venv --system-site-packages env
	source ~/env/bin/activate
	pip install ipykernel==5.1.1 --no-deps
	pip install -r ~/ska-sdc-2/environment/requirements.txt
	module load jupyter-utils
	kernel-create -f -n env
	cp ~/ska-sdc-2/environment/launcher ~/.local/share/jupyter/kernels/env/launcher
	cp ~/ska-sdc-2/environment/.jupyterhub.env .jupyterhub.env
elif [ "$resetenv" = true ]
then
	rm -r ~/env
	
	if [ -d ~/.local/lib/python3.8 ]
	then
		rm -r ~/.local/lib/python3.8
	fi
	
	source ~/ska-sdc-2/environment/daint.bash
	source ~/env/bin/activate
else
	source ~/env/bin/activate
fi