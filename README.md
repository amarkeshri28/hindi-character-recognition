# hindi-character-recognition
The folder should be downloaded as it is, and the  model file must be downloaded from the models folder from this link - https://drive.google.com/drive/folders/1obh5CUAfGqmBcdlAepBP4uUrE-KqWtcJ?usp=sharing. Save the model .h5 file in the environment folder.

## Dataset

+ The dataset can be found here https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset

## File Description

Edit files and execute files in the following order and change paths where ever necessary
+ main.py - code of segmentation and character recognition 
+ train.ipynb - trainer code of model
+ character.txt - contain characters which are segmented and recognised correctly

#### Important
+ Don't leave space between hindi words and it's above connecting line if a word contains more than 1 character
+  Some times predicted data can be shuffled i.e  not in correct order
+  Most data is tested on words written on notepad, with different noises as hand written with pen or sketch pen can be very flawed but still the model performs pretty well on handwritten data too 

#### Additional features
We have created as well as trained the model on a dataset containing ***VOWELS*** as well as ***CONSONANTS***
+ Noise removal
+ Reading sparse as well as connected characters
+ Can deal on line noises/circular noises or noise of any category
+ Can Handle image of any size
+ Can detect characters from multiple lines
+ Dufferent types of cases our model can handle are given in sampleImages folder

### Architecture
##### CONV2D --> MAXPOOL --> CONV2D --> MAXPOOL -->FC -->Softmax--> Classification


### Python Implementation
##### 1. Dataset- Devnagari Character Dataset.
##### 2. Images of size 32 X 32.
##### 3. Convolutional Network (CNN).

------------

### Train Acuracy ~ 98%
### Test Acuracy ~ 94%
