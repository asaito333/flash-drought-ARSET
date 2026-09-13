conda env create -f environment.yml
if ($LASTEXITCODE -ne 0) { conda env update --prune -f environment.yml }
(& conda "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate flash-drought
python -m ipykernel install --user --name=flash-drought --display-name "Python (flash-drought)"
jupyter lab notebooks\1_drought_indicators.ipynb