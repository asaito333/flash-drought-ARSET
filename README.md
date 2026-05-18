# Flash Drought NASA ARSET Course
### Agricultural Flash Drought Detection Using Solar-Induced Fluorescence (SIF) and Soil Moisture Data

This repository contains code the NASA ARSET course on Agricultural Flash Drought. The training is comprised of two exercises:

1. **1_drought_indicators.ipynb**: Methods for retrieving SIF and Soil Moisture data from the OCO-2 and SMAP missions, and deriving time series of the Rapid Change Index (RCI) based on these measurements. SIF-RCI and Soil Water Deficit Index (SWDI) will be used to detect flash droughts in two real-world scenarios.
2. **2_detection.ipynb**: We will apply the indicators we derived in the first exercise to build a framework for detecting flash droughts in general. The effectiveness of different definitions of flash drought will be compared.

## Learning Objectives

By the end of this course, you will learn how to:

* Identify the various remote sensing data sources used for the detection and prediction of flash drought events.
* Compare the advantages and limitations of different measurements to describe the ecological impacts of flash drought.
* Use a Jupyter Notebook to generate a time series of flash drought detection indices using Solar-Induced Fluorescence (SIF) and Soil Moisture (SM) data.
* Compare the effectiveness of different heuristics applied to flash drought detection indices in real-world scenarios.
* Identify historical flash drought occurrences in regions around the world by generalizing the data-driven approach outlined in the Jupyter notebook exercises.

## Contact

Please email [Jacqueline Ryan](mailto:Jacqueline.Ryan@jpl.nasa.gov) at JPL for any questions about the code in this course.

## Citations

[1] Lisonbee, J., Woloszyn, M., Skumanich, M. (2021). Making sense of flash drought:
definitions, indicators, and where we go from here. Journal of Applied and Service Climatology, 1, https://doi.org/10.46275/JOASC.2021.02.001

[2] Christian, J. I., Hobbins, M., Hoell, A., Otkin, J. A., Ford, T. W., Cravens, A. E., Powlen, K. A., Wang, H., & Mishra, V. (2024). Flash drought: A state of the science review. WIREs Water, 11(3), e1714, https://doi.org/10.1002/wat2.1714

[3] Mohammadi, K., Jiang, Y. & Wang, G. (2022). Flash drought early warning based on the trajectory of solar-induced chlorophyll fluorescence, Proc. Natl. Acad. Sci. U.S.A. 119 (32) e2202767119, https://doi.org/10.1073/pnas.2202767119

[4] Mohammadi, K., & Wang, G. (2025). Impact Matters: Detection and Early Warning of Agriculturally Impactful Flash Droughts. Bull. Amer. Meteor. Soc., 106, E752–E769, https://doi.org/10.1175/BAMS-D-24-0143.1

[5] He, M., Kimball, J. S., Yi, Y., et al. (2019). Impacts of the 2017 flash drought in the US Northern plains informed by satellite-based evapotranspiration and solar-induced fluorescence. Environ. Res. Lett., 14, https://doi.org/10.1088/1748-9326/ab22c3

[6] Kimball, J. S., Jones, L., Jensco, K., He, M., Maneta, M. P., & Reichle, R. H. (2019). SMAP L4 Assessment of the US Northern Plains 2017 Flash Drought. International Geoscience and Remote Sensing Symposium (IGARSS), 5366-5369.

[7] Sehgal, V., Gaur, N., & Mohanty, B. P. (2021). Global flash drought monitoring using surface soil moisture. Water Resources Research, 57, e2021WR029901, https://doi.org/10.1029/2021WR029901

[8] Brust, C., Kimball, J. S., Maneta, M. P., Jencso, K., He, M., & Reichle, R. H. (2021).
Using SMAP Level-4 soil moisture to constrain MOD16 evapotranspiration over the contiguous USA. Remote Sensing of Environment, 255, 112277, https://doi.org/10.1016/j.rse.2020.112277

[9] Tang, S., Wang, S., Jiang, J., & Zheng, Y. (2026). Improved flash drought forecasting and attribution: A spatial-temporal causality-aware deep learning approach. Journal of Hydrology, 667, 134945, https://doi.org/10.1016/j.jhydrol.2026.134945

[10] Li, X. and Xiao, J. (2019). A global, 0.05-degree product of solar-induced chlorophyll fluorescence derived from OCO-2, MODIS, and reanalysis data. Remote Sensing, 11, 517. https://doi.org/10.3390/rs11050517

[11] Reichle, R., De Lannoy, G., Koster, R. D., Crow, W. T., Kimball, J. S., Liu, Q. & Bechtold, M. (2025). SMAP L4 Global 3-hourly 9 km EASE-Grid Surface and Root Zone Soil Moisture Geophysical Data. (SPL4SMGP, Version 8). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/T5RUATAQREF8


## Data Sources and Attributions

**GOSIF Dataset:** Used with permission from the author.

**All Code:** Copyright 2026, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
