#!/bin/bash
conda env create -f environment.yml 2>/dev/null || conda env update --prune -f environment.yml
eval "$(conda shell.bash hook)"
conda activate flash-drought
python -m ipykernel install --user --name=flash-drought --display-name "Python (flash-drought)"
jupyter lab notebooks/1_drought_indicators.ipynb