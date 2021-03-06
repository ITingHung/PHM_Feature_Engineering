# PHM Feature Engineering
## Introduction
Prognostics and Health Management (PHM) is an essential topic in both academia and industry. In order to extract useful information from data, feature engineering is often applied to derive some features. Some common features are organized in [feature_engineering](https://github.com/ITingHung/PHM_Feature_Engineering/tree/main/feature_engineering), which includes both time and frequency domain. 

To display how to utilize the package, an example is demonstrated by data from [FEMTO Bearing Data Set](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/). For more detail about the dataset, please refer to [IEEE PHM 2012 Prognostic challenge](https://github.com/ITingHung/PHM_Feature_Engineering/blob/main/FEMTOBearing/IEEEPHM2012-Challenge-Details.pdf).

## Data Format
Below shows the format of input and output data for feature engineering. By inputting a dataset with series of samples, some time domain and frequency domain features will be derived for each column in the original dataset.
<p align="center">
<img src="./image/Input Output Illusration.png" alt="Input & Output Format Illusration" title="Input & Output Format Illusration" width="500">
</p>


