#!/bin/bash
python3 -m venv flash-drought
source flash-drought/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m ipykernel install --user --name=flash-drought --display-name "Python (flash-drought)"
jupyter lab notebooks/1_drought_indicators.ipynb