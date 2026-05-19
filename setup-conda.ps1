conda env create -f environment.yml
conda activate flash-drought
python -m ipykernel install --user --name=flash-drought --display-name "Python (flash-drought)"
jupyter lab notebooks\1_drought_indicators.ipynb