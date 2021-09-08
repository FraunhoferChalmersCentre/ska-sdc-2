This is the repository for team FORSKA-Sweden participating in SKA Science Data Challenge 2.

## Requirements
Linux is the only OS used in development. CUDA-compatible GPU is highly recommended for acceleration of training and
traversing tasks.

## Installation

If you are using Swiss supercomputer Piz Daint (https://user.cscs.ch/), simply run `source environment/daint.bash`.

Otherwise, do the following:

`pip install environment/requirements.txt`

You also need to install also the correct version of CUDA toolkit, depending on hardware. Find out more about
it here: https://pytorch.org/get-started/previous-versions/

Further, also SoFiA 1.3.2 (see https://github.com/SoFiA-Admin/SoFiA for more information) needs to be installed into the
environment:

`pip install https://github.com/SoFiA-Admin/SoFiA/archive/refs/tags/v1.3.2.tar.gz`

## Instructions
A typical use of the code may look like this:
1. Download data files: `python download_data.py --type dev_s`. Available data types are dev_s, dev_l & eval.
2. Create data set files: `python create_dataset.py`. Will create datset files to be used in model fitting.
3. Train the model: `python model_fitting.py`. Models performing best validation will be saved to /saved_models/
4. Select saved model to use by changing traversing -> checkpoint in config.yaml
5. Create data for hyperparameter optimization: `python save_hparam_dataset.py`
6. Run hyperparameter optimization: `python hyperparameter_search.py`
7. Currently, hyperparameters need to be set manually. Hyperparameters and corresponding scores are logged to Tensorboard in hparam_logs directory.
8. Traversing a HI cube (fits-file) to produce a full catalogue: `python traverse_cube.py`. Argument `--n-parallel` can
be used to split the job into `n-parallel` separate and independent jobs. Also specify 0<_`--i-job`<`n-parallel`, to 
specify index of the job to start.
9. If traversing was split to different jobs, there will be a separate catalogue for each job. To merge these into one
single file: `python merge_catalogues.py`

## Useful links:

Web page for SDC2: https://sdc2.astronomers.skatelescope.org/

Discussion and support forum: https://sdc2-discussion.astronomers.skatelescope.org/