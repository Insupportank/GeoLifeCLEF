# Video link
https://www.youtube.com/watch?v=9CTslkW2xLs

# Template base code for pytorch

This repository contains a template base code for a complete pytorch pipeline.

This is a template because it works on fake data but aims to illustrate some pythonic syntax allowing the pipeline to be modular.

More specifically, this template base code aims to target :

- modularity : allowing to change models/optimizers/ .. and their hyperparameters easily
- reproducibility : saving the commit id, ensuring every run saves its assets in a different directory, recording a summary card for every experiment, building a virtual environnement

For the last point, if you ever got a good model as an orphane pytorch tensor whithout being able to remember in which conditions, with which parameters and so on you got, you see what I mean. 

## Usage

### Local experimentation

For a local experimentation, you start by setting up the environment :

```
python3 -m virtualenv venv
source venv/bin/activate
python -m pip install .
```

Then you can run a training, by editing the yaml file, then 

```
python -m torchtmpl.main config.yaml train
or on slurm job:
python3 submit-slurm.py config.yaml 1
```

And for testing (making sample submission)

```
python -m torchtmpl.main config.yaml test
copy the file to local computor:
scp -r sdi-labworks-2023-2024_XXX@chome.metz.supelec.fr:~/Documents/SDI/Deep_Learning/deep_learning_geolifeclef_aaron_julien_olivier_2024/sample_submission_XXX.csv .
```

## Testing the functions

Every module/script is equiped with some test functions. Although these are not unitary tests per se, they nonetheless illustrate how to test the provided functions.

For example, you can call :


```
python3 -m virtualenv venv
source venv/bin/activate
python -m pip install .
python -m torchtmpl.models
```

and this will call the test functions in the `torchtmpl/models/__main__.py` script.

