if [ ! -d ~/scratch ]; then
	ln -s $SCRATCH scratch
fi

module load daint-gpu
module load PyTorch/1.6.0
module load Horovod/0.19.5-CrayGNU-20.08-pt-1.6.0

if [ ! -d ~/env ]; then
	python -m venv --system-site-packages env
	source ~/env/bin/activate
	pip install ipykernel==5.1.1 --no-deps
	pip install -r ~/ska-sdc-2/environment/requirements.txt
	module load jupyter-utils
	kernel-create -f -n env
	cp ~/ska-sdc-2/environment/launcher ~/.local/share/jupyter/kernels/env/launcher
	cp ~/ska-sdc-2/environment/.jupyterhub.env .jupyterhub.env
else
	source ~/env/bin/activate
fi