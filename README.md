# SCMI30-IITRPR: Smartphone Camera Model Identification Dataset Comprising both Similar and Random Content Image Sets

  SCMI30-IITRPR contains images taken from 30 smartphone cameras, of various renowned brands like Vivo, Oppo, Realme, Samsung, OnePlus, Nothing, Poco, Motorola, Redmi, and Apple. The dataset covers a broad spectrum of smartphone costs, catering to diverse demographic groups. The devices also have different operating systems. Images are captured in default auto-settings, ensuring consistent focus, white balance, and High Dynamic Range (HDR). All the images are saved in the existing jpg format.

## Declaration

### Declaration of competing interest
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. 

### Declaration of Code/ Reproducibility
All the prior CMI methods are self implemented using pytorch framework and python.

## How to run the codes

### Prepare the environment

`pip install requirements.txt`

This will install all the dependencies needed to run the codes. It is highly recommended that you create a new environment when trying out the codes.

You can create a new conda enviromnent using `conda create --name <env_name> python=<version>`. 

For eg: `conda create --name scmi30test python=3.11`

### Create train test split of the data

Use the csv files present at [Link to Dataset](https://www.kaggle.com/dsv/7589186) to create the train and test splits. The image labels used for creating the test split is present for random, similar and merged. Merged split is essentially combination of random and similar splits. All the data is individually 80:20 splitted among the classes to maintain equal representation of the classes in train and test sets.

### How to split folders should look like

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

### Command for generating patches

`python <patch_script.py> path/to/base/dir/having/train/test/split`

For eg: If the base dir having the train test split is named: `data_split`

Run: `python rana_patches.py data_split/`

All the codes for generation of patches can be found inside Patches_Codes

### Trained Model weights

You can download the trained model weights by using the google drive link: [Google Drive](https://drive.google.com/drive/folders/1Fp242mDkF5BjmKLC-8W19M3sUwIwCQvz?usp=sharing)

### Testing out different methods

To test any method presented in the paper use the following commands.

`python <method>_test.py --data path/to/patch/data --model path/to/trained/model`

This will start the python script for generating the results. One can use different combinations to generate the entire results table present in the paper.

Note: To run `chen_test.py`, no need to pass the patched data. Directly pass the test folder of either `random`, `similar` or `merged`

Example: `python chen_test.py --data /data/similar/test/ --model chen_`

## Acknowledgements
We thank our colleagues from the CSE Department, Indian Institute of Technology Ropar for their time and support during the creation of the SCMI30-IITRPR dataset.

## Citation

```bibtex
@misc{kapil rana_abhilasha s jadhav_protyay dey_vishwas rathi_puneet goyal_gaurav sharma_2024,
  	title={SCMI30-IITRPR},
  	url={https://www.kaggle.com/dsv/7589186},
  	DOI={10.34740/KAGGLE/DSV/7589186},
  	publisher={Kaggle},
  	author={Kapil Rana and Abhilasha S Jadhav and Protyay Dey and Vishwas Rathi and Puneet Goyal and Gaurav Sharma},
  	year={2024}
}
```

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/IPSA-Lab/scmi30-iitrpr/blob/main/LICENSE)

## Authors

* Kapil Rana - Computer Science and Engineering, Indian Institute of Technology Ropar, Punjab, India
* Protyay Dey - Computer Science and Engineering, Indian Institute of Technology Ropar, Punjab, India
* Abhilasha S. Jadhav - Computer Science and Engineering, Indian Institute of Technology Ropar, Punjab, India
* Vishwas Rathi - Computer Science and Engineering, National Institute of Technology Kurukshetra, Haryana, India
* Puneet Goyal - Computer Science and Engineering, Indian Institute of Technology Ropar, Punjab, India
* Gaurav Sharma - Electrical and Computer Engineering, University of Rochester, Rochester, New York, USA




