# SCMI30-IITRPR: Smartphone Camera Model Identification Dataset Comprising both Similar and Random Content Image Sets

  SCMI30-IITRPR contains images taken from 30 smartphone cameras, of various renowned brands like Vivo, Oppo, Realme, Samsung, OnePlus, Nothing, Poco, Motorola, Redmi, and Apple. The dataset covers a broad spectrum of smartphone costs, catering to diverse demographic groups. The devices also have different operating systems. Images are captured in default auto-settings, ensuring consistent focus, white balance, and High Dynamic Range (HDR). All the images are saved in the existing jpg format.

## Table Of Contents
* [How to run the codes](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Dataset Description](#dataset-description)
* [License](#license)
* [Authors](#authors)
* [Acknowledgements](#acknowledgements)

## How to run the codes

Step 1: `pip install requirements.txt`

This will install all the dependencies needed to run the codes. It is highly recommended that you create a new environment when trying out the codes.

You can create a new conda enviromnent using `conda create --name <env_name> python=<version>`. 

For eg: `conda create --name scmi30test python=3.11`

### Create train test split of the data



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

## Dataset Description

### **General Description:**
The SCMI30-IITRPR contains images taken from 30 smartphone cameras, of various renowned brands like Vivo, Oppo, Realme, Samsung, OnePlus, Nothing, Poco, Motorola, Redmi, and Apple. The details of the devices are given below:

| SNo. | Camera Id                  | Brand    | Model           | OS Version       | Image Resolution |
| ---- | -------------------------- | -------- | --------------- | ---------------- | ---------------- |
| 1    | D01_Samsung_Galaxy_S20Plus | Samsung  | Galaxy S20 Plus | Android 13       | 4032 x 2268      |
| 2    | D02_Nothing_One            | Nothing  | One             | Nothing OS 1.5.6 | 4096 x 3072      |
| 3    | D03_Samsung_Galaxy_A03     | Samsung  | Galaxy A03      | Android 12       | 4000 x 3000      |
| 4    | D04_Samsung_Galaxy_M04     | Samsung  | Galaxy M04      | Android 12       | 4160 x 3120      |
| 5    | D05_Vivo_V9_Pro            | Vivo     | V9 Pro          | Android 9        | 4160 x 3120      |
| 6    | D06_Apple_Iphone_12Mini    | Apple    | Iphone 12 Mini  | iOS 16.2         | 4032 x 3024      |
| 7    | D07_Apple_Iphone_11        | Apple    | Iphone 11       | iOS 16.6         | 4032 x 3024      |
| 8    | D08_Redmi_Note_8Pro        | Redmi    | Note 8 Pro      | Android 9        | 4624 x 3472      |
| 9    | D09_Samsung_Galaxy_J8_10G  | Samsung  | Galaxy J810G    | Android 10       | 4608 x 2592      |
| 10   | D10_Samsung_Galaxy_F41     | Samsung  | Galaxy F41      | Android 12       | 4624 x 2136      |
| 11   | D11_OnePlus_8T             | OnePlus  | 8T              | Android 12       | 4000 x 1800      |
| 12   | D12_Vivo_Y02t              | Vivo     | Y02t            | Android 13       | 3264 x 1836      |
| 13   | D13_Oppo_A17k              | Oppo     | A17k            | Android 12       | 3264 x 1840      |
| 14   | D14_Samsung_S20FE          | Samsung  | S20 FE          | Android 13       | 4000 x 3000      |
| 15   | D15_Motorola_Motog16       | Motorola | Moto G60        | Android 12       | 4000 x 3000      |
| 16   | D16_Samsung_Galaxy_S21FE   | Samsung  | Galaxy S21 FE   | Android 14       | 4000 x 3000      |
| 17   | D17_Apple_Iphone_12        | Apple    | Iphone 12       | iOS 17.2         | 4032 x 3024      |
| 18   | D18_IQOO_Z3                | IQOO     | Z3 5G           | Android 13       | 4608 x 3456      |
| 19   | D19_IQOO_Z6_Lite           | IQOO     | Z6 Lite         | Android 13       | 4080 x 3060      |
| 20   | D20_Motorola_MotoG73_5G    | Motorola | G73 5G          | Android 13       | 4096 x 3072      |
| 21   | D21_OnePlus_10Pro_5G       | OnePlus  | 10 Pro 5G       | Android 13       | 4000 x 2252      |
| 22   | D22_Poco_F5                | Poco     | F5              | Android 13       | 4624 x 3472      |
| 23   | D23_Poco_F5_Pro            | Poco     | F5 Pro          | Android 13       | 4000 x 3000      |
| 24   | D24_Realme_8               | Realme   | 8               | Android 13       | 4624 x 3468      |
| 25   | D25_Realme_X3_Superzoom    | Realme   | X3 Superzoom    | Android 12       | 4608 x 3456      |
| 26   | D26_Redmi_9i_Sport         | Redmi    | 9i Sport        | Android 10       | 4160 x 3120      |
| 27   | D27_Redmi_Note10_Pro       | Redmi    | Note 10 Pro     | Android 12       | 4640 x 3072      |
| 28   | D28_Apple_Iphone_13        | Apple    | Iphone 13       | iOS 17.2         | 4032 x 3024      |
| 29   | D29_Apple_Iphone_15        | Apple    | Iphone 15       | iOS 17.2         | 4032 x 3024      |
| 30   | D30_Vivo_Y75               | Vivo     | Y75 5G          | Android 13       | 4080 x 3060      |


The dataset covers a broad spectrum of smartphone costs, catering to diverse demographic groups. The devices also have different operating systems. Images are captured in default auto-settings, ensuring consistent focus, white balance, and High Dynamic Range (HDR). All the images are saved in the existing jpg format.

### **Categorical Description:**

#### **Type1 (Random):**

These images are randomly taken by the device owner of the devices, depicting various objects and landscapes of various lighting conditions, including daytime, fog, and nighttime. A minimum 150 number of images have been captured using each device which results in 5287 images in total in the Type1 category.

#### **Type2 (Similar):**

These images were captured in a controlled environment, with a fixed set of photos taken by a single user while the device owner observed. This category encompasses the following image types per set:
- Texture Images: 42 images per device
- Object Images: 44 images per device
- Color Palettes: 15 images per device
- No Content Images: 2 images (Black and Wall) per device
- Natural Images: 52 images per device

#### **Dataset Characteristics:**

- Total Images: 5287 (Random) + 4650 (Similar)
- File Format: .jpg (Inbuilt Quality of smartphone camera), Ex: D01_nat_1.jpg.
- Resolution: Varies across devices (specified earlier)
- Metadata: Limited to camera settings and device information

#### **Dataset Structure:**

The SCIM30-IITRPR folder contains 2 Folders (Random and Similar) and 3 CSV files.
Random folder contains folders for 30 devices named D01, D02, …, and D30 where each contains at least 150 random images taken by the owner. The Similar folder also contains folders for 30 devices named D01, D02, …, and D30 where each folder contains 5 subfolders: Natural, Objects, Colors, Textures, and No Content containing the corresponding number of images mentioned earlier.

For test images of Random, Similar and Merged(Similar + Random) sets refer to: random_test_set.csv, similar_test_set.csv and merged_test_set.csv files respectively. 

#### **Naming Conventions:**

The nomenclature used in Random folder is as follows: Devices are named as De
vice number followed by smartphone brand followed by smartphone model. For instance D02 Nothing One, D24 Realme 8 etc. The images in each folder are randomly clicked so we named them as device number followed by “rnd” followed by image number. Ex: D01 rnd 1.jpg, D01 rnd 2.jpg etc. The Similar folder consists of 30 smartphone subfolders. Each of the smartphone folders present in the Similar folder consists of five more subfolders namely “colors”, “natural”, “no_content”, “objects” and “texture”. The nomenclature followed in these subfolders is as follows: The images present in the colors subfolder is named as device number followed by “color” followed by image number. Ex D02 color 1.jpg, D03 color 3.jpg etc respectively. Similarly for the natural, objects and texture folders
The naming convention is D02 nat 1.jpg, D02 obj 1.jpg, D02 tex 1.jpg etc. The no content subfolder consists of two images Dx black.jpg and Dx wall.jpg, x corresponds to the device number in which these images are present.
Usage:
The SCMI30-IITRPR dataset must be used for research and education purposes only. It can be beneficial for tasks like image classification, object recognition, camera model identification, etc. Moreover, the dataset enables comparison between different smartphone cameras and facilitates the development of algorithms to optimize device recognition. This dataset is designed in such a way that it covers a wide range of images for intensive image training of the model.

## Acknowledgements
We thank our colleagues from the CSE Department, Indian Institute of Technology Ropar for their time and support during the creation of the SCMI30-IITRPR dataset.

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



