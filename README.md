# Benchmarking Code for SCMI30-IITRPR: Smartphone Camera Model Identification Dataset Comprising both Similar and Random Content Image Sets

  This repository provides code and trained models for reproducing the results presented in the paper "SCMI30-IITRPR: Smartphone Camera Model Identification Dataset Comprising both Similar and Random Image Sets,'' K. Rana, A. Jadhav, P. Dey, V. Rathi, P. Goyal, and G. Sharma, submitted for review. Users are referred to the paper for details of the benchmarks. Researchers that make use of the code should cite the paper.

## Declaration

### Declaration of competing interest
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. 

### Declaration of Code/ Reproducibility
All the prior CMI methods are self implemented using pytorch framework and python.

## Dataset Description

For the detailed overview of the dataset kindly refer [Dataset Overview]( https://doi.org/10.34740/kaggle/dsv/7589186)

## Intended Usage
> [!Note]
> The SCMI30-IITRPR dataset must be used for research and education purposes only. The dataset is designed primarily to facilitate research on camera model identification (CMI) but could also be beneficial for other tasks such as individual camera device forensics, image classification, and object recognition. 
> Researchers that make use of the dataset should cite the [dataset](#cite) and the associated paper ``SCMI30-IITRPR: Smartphone Camera Model Identification Dataset Comprising both Similar and Random Image Sets,'' K. Rana, A. Jadhav, P. Dey, V. Rathi, P. Goyal, and G. Sharma, submitted for review.

## Steps to run the codes

### Preparing the environment

`pip install -r requirements.txt`

This will install all the dependencies needed to run the codes. 

> [!Tip]
> It is highly recommended that we create a new environment when executing these codes.

New conda enviromnent can be created using `conda create --name <env_name> python=<version>`. 

For eg: `conda create --name scmi30test python=3.11`

### Creating train test split of the data

Using the csv files present at [Link to Dataset](https://doi.org/10.34740/kaggle/dsv/7589186) train and test splits can be created. The image labels used for creating the test split is present for random, similar and merged. Merged split is essentially combination of random and similar splits. All the data is individually 80:20 splitted among the classes to maintain equal representation of the classes in train and test sets.

### Directory structure after splitting the data

<pre>
data_split
|-- random
|   |-- train
|   |   |-- D01_Samsung_Galaxy_S20Plus
|   |   |-- ...
|   |   |-- D30_Vivo_Y75
|   |-- test
|       |-- D01_Samsung_Galaxy_S20Plus
|       |-- ...
|       |-- D30_Vivo_Y75
|-- similar
|   |-- train
|   |   |-- D01_Samsung_Galaxy_S20Plus
|   |   |-- ...
|   |   |-- D30_Vivo_Y75
|   |-- test
|       |-- D01_Samsung_Galaxy_S20Plus
|       |-- ...
|       |-- D30_Vivo_Y75
|-- merged
    |-- train
    |   |-- D01_Samsung_Galaxy_S20Plus
    |   |-- ...
    |   |-- D30_Vivo_Y75
    |-- test
        |-- D01_Samsung_Galaxy_S20Plus
        |-- ...
        |-- D30_Vivo_Y75
</pre>

### Script for generating patches

`python <patch_script.py> path/to/base/dir/having/train/test/split`

For eg: If the base dir having the train test split is named: `data_split`

Run: `python rana_patches.py data_split/`

All the codes for generation of patches can be found in `Patches_Codes` folder.

### Trained Model weights

The trained model weights can be downloaded using the google drive link: [Google Drive](https://drive.google.com/drive/folders/1Fp242mDkF5BjmKLC-8W19M3sUwIwCQvz?usp=sharing)

### Testing different methods

To test the models following commands can be used.

Different methods available:
* [[1]](#ref1) Rana et. al
* [[2]](#ref2) Rafi et. al
* [[3]](#ref3) Chen et. al 
* [[4]](#ref4) Liu et. al 
* [[5]](#ref5) Bennabhaktula et. al

`python <method>_test.py --data path/to/patch/data --model path/to/trained/model`

This python script will generate results corresponding to respective trained models.

>[!Note]
>To run `chen_test.py`, we don't have to pass the patched data. We can straight away pass the test folder of either `random`, `similar` or `merged`. <br>
>Example: `python chen_test.py --data /data/similar/test/ --model Chen_trained_model_random_final`

## Acknowledgements
We thank our colleagues from the CSE Department, Indian Institute of Technology Ropar for their time and support during the creation of the SCMI30-IITRPR dataset.

## Citation 
<a name="cite"></a>
```bibtex
@misc{kapil_rana_abhilasha_s_jadhav_protyay_dey_vishwas_rathi_puneet_goyal_gaurav_sharma_2024,
	title={SCMI30-IITRPR},
	url={https://www.kaggle.com/dsv/7589186},
	DOI={10.34740/KAGGLE/DSV/7589186},
	publisher={Kaggle},
	author={Kapil Rana and Abhilasha S Jadhav and Protyay Dey and Vishwas Rathi and Puneet Goyal and Gaurav Sharma},
	year={2024}
}
```

## License
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/IPSA-Lab/scmi30-iitrpr/blob/main/LICENSE)

## Disclaimer

The code is provided "as is" with ABSOLUTELY NO WARRANTY expressed or implied. Use at your own risk.

## Authors

* Kapil Rana - Computer Science and Engineering, Indian Institute of Technology Ropar, Punjab, India
* Protyay Dey - Computer Science and Engineering, Indian Institute of Technology Ropar, Punjab, India
* Abhilasha S. Jadhav - Computer Science and Engineering, Indian Institute of Technology Ropar, Punjab, India
* Vishwas Rathi - Computer Science and Engineering, National Institute of Technology Kurukshetra, Haryana, India
* Puneet Goyal - Computer Science and Engineering, Indian Institute of Technology Ropar, Punjab, India
* Gaurav Sharma - Electrical and Computer Engineering, University of Rochester, Rochester, New York, USA

## References

* [Kapil Rana, Puneet Goyal, and Gaurav Sharma. "Dual-branch convolutional neural network for robust camera model identification." Expert Systems with Applications, vol. 238, pp. 121828, 2024](https://doi.org/10.1016/j.eswa.2023.121828) <a name="ref1"></a>

* [Abdul Muntakim Rafi, Thamidul Islam Tonmoy, Uday Kamal, QM Jonathan Wu, and Md Kamrul Hasan. "Remnet: Remnant convolutional neural network for camera model identification." Neural Computing and Applications, vol. 33, pp. 3655–3670, 2021](https://doi.org/10.1007/s00521-020-05220-y) <a name="ref2"></a>

* [Yunxia Liu, Zeyu Zou, Yang Yang, Ngai-Fong Bonnie Law, and Anil Anthony Bharath. "Efficient source camera identification with diversity-enhanced patch selection and deep residual prediction." Sensors, vol. 21, no. 14, pp. 4701, 2021](https://doi.org/10.3390/s21144701) <a name="ref3"></a>

* [Yunshu Chen, Yue Huang, and Xinghao Ding. "Camera model identification with residual neural network." in 2017 IEEE International Conference on Image Processing (ICIP). IEEE, 2017, pp. 4337–4341](https://doi.org/10.1109/ICIP.2017.8297101) <a name="ref4"></a>

* [Guru Swaroop Bennabhaktula, Enrique Alegre, Dimka Karastoyanova, and George Azzopardi. "Camera model identification based on forensic traces extracted from homogeneous patches." Expert Systems with Applications, vol. 206, pp. 117769, 2022](https://doi.org/10.1016/j.eswa.2022.117769) <a name="ref5"></a>




