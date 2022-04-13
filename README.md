# Document Classification using Transformers

## Setup
Prerequisites: pyenv

1. Install python version: ``pyenv install 3.9.9``
1. Set pythont to belocal in this folder ``pyenv local 3.9.9``
1. Create new pipenv: ``pipenv --python $(pyenv which python)``
1. Activate pipenv:
1. Install requirements: 
1. Install english spacy: ``python -m spacy download en``


`conda env export --no-builds > env.yml`
## Conda
1. `conda env create -f env.yml -n dissertation python=3.9.1`
## Run
`` python main.py --config config/dev.json --eval data/processed/fce/json/dev_small.json --train data/processed/fce/json/dev_small.json
``