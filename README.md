# Document Classification using Transformers

## Setup
Prerequisites: Anaconda

1. Set up conda env from the env.yml file.
1. Install english spacy: ``python -m spacy download en``


## Conda
To dump the environment: `conda env export --no-builds > env.yml`

Then, can recreate it somewhere else using: `conda env create -f env.yml -n dissertation python=3.9.1`

## Run
`` python main.py --config config/dev.json --eval data/processed/fce/json/dev_small.json --train data/processed/fce/json/dev_small.json
``

## Slurm
Change config and datasets in the `run_slurm.sh` file

Submit job: `sbatch run_slurm.sh`

## Explore BeerAdvocate data
``python processed/beer_advocate/explore_dist.py -input_json original/beer_advocate/annotations.json``