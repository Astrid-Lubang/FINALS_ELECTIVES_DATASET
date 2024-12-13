# TABLE OF CONTENTS 

* [I. INTRODUCTION](#i-introduction)
* [II. ABSTRACT](#ii-abstract) 
* [III. PROJECT METHODS](#iii-projectmethods) 
* [IV. CONCLUSIONS](#iv-conclusions) 
* [V. ADDITIONAL MATERIALS](#v-additionalmaterials) 
* [VI. REFERENCES](#vi-references) 



# I. INTRODUCTION 

<p align="justify"> 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;OpenCV (Open Source Computer Vision Library) containing more-than 2500 optimized algorithms that can be used for many different computer vision applications such as face detection and recognition, objective identification, object tracking, image registration and stitching. Cristina (2024). When it comes to identifying dog breeds and overlaying their names on images it addresses the accurately labeling and classifying different dog breeds in real time. With over 300 recognized dog breeds globally, each  breeds has a unique features, identifying dog breeds can be challenging.  But openCV addresses it because it provides tools for images enabling the developers to detect, recognize and process visual effectively. This issue is particularly relevant  in animal adoption centers,dog training schools, and veterinary clinics, wherein they can correctly identify the dog breeds. <br>

<p align="justify"> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The significance of this projects is to make less complex in breed identification using computer vision and machine learning techniques.In this project the system can save time, improve accuracy and reduce human error.It demonstrates the importance and efficiency of openCV in solving real world problems.<br>
 
# II. ABSTRACT

<p align="justify"> 

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The main objective of this project is to identifying dog breeds by overlaying breed names on their images with the help of openCV.This approach demonstrates the practical application of computer vision in simplifying breed recognition, offering a user-friendly solution for pet owners, shelters, and veterinary clinics. <br>
 
<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The following are the specific objectives that this project aims to achieve: <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1. To develop a system that is capable of accurately  identifying dog breeds from images using computer vision. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2. To implement functionality in overlay the identified dog breeds name directly on the dogs image.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3. To create an intuitive  interface for users, enabling easy uploads and display results<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.: To optimize the system to handle minimal latency and high accuracy.<br>

<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The following is the approach to achieve the project objectives: <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1. Gather dataset of dog images  representing  a wide variety of dogs.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2. Model selection and training for image classification and pre trained models.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3. Use openCV to process inputs and detect the dogs images.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4. Testing and optimization to ensure the high accuracy of the system<br>

<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The following are the expected outputs of the project:<br>
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1. A fully functional system capable of identifying dog breed images with high accuracy of 80-90%.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2. Overlaying of breed names on processed images.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3. A user friendly tool that identify the dog breeds for variety applications, such as animal shelters, pet owners and veterinary.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4. A demonstration of how OpenCV and machine learning can be integrated to solve practical problems efficiently.<br>

# III. PROJECT METHODS

<p align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	The methodology  is how our project will be made .It should be focused on the step by step process in identifying dog breeds by overlaying dog breeds name on their images. <br>



&nbsp;&nbsp;&nbsp;&nbsp; **Dataset Preparation and library installation**
- Cloned the GitHub repository containing the dog breed dataset.
- Structured the data into organized folders based on dog breeds.
- Installed essential libraries: TensorFlow, OpenCV, and Matplotlib.

&nbsp;&nbsp;&nbsp;&nbsp; **Image Visualization**
- Used OpenCV to read images and Matplotlib for displaying them.
- Extracted breed names from filenames for annotated visualization.

&nbsp;&nbsp;&nbsp;&nbsp; **Model Development**
- Built a model using MobileNetV2 as the base.
- Added custom layers for classification and softmax output.
- Compiled the model with the Adam optimizer and categorical cross-entropy loss.

&nbsp;&nbsp;&nbsp;&nbsp; **Enhanced Model Training Pipeline**
- Trained the model on the prepared dataset for 5 epochs.
- Validated performance and ensured generalization.
- Developed functions for image preprocessing and breed prediction.
- Displayed test images with annotated predictions using Matplotlib.

 &nbsp;&nbsp;&nbsp;&nbsp; **Evaluation**
- Tested the model on unseen images.
- Visualized classification results with accurate breed labels.<br>

# IV. CONCLUSIONS
<p align="justify"> 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This project demonstrates a dog breed classification pipeline using transfer learning with MobileNetV2. It involves data preprocessing, model training, and displaying predictions with breed names overlayed on images. The approach leverages a pre-trained model to achieve high accuracy with limited data and computational resources.

&nbsp;&nbsp;&nbsp;&nbsp; **Findings**
- **Data Preprocessing**: Normalization and augmentation for robust training.
- **Transfer Learning**: MobileNetV2 as the base model.
- **Breed Prediction**: Classifies and overlays the predicted breed on dog images.
- **Custom Visualization**: Flexible text positioning for breed names on images.

&nbsp;&nbsp;&nbsp;&nbsp; **Challenges**
- Dataset organization and filename inconsistencies.
- Addressing overfitting due to limited data.
- Balancing training speed and accuracy through hyperparameter tuning.

&nbsp;&nbsp;&nbsp;&nbsp; **Outcomes**
- Successfully implemented a dog breed classifier with high accuracy.
- Validated the effectiveness of transfer learning for image classification tasks.
- Produced a visually informative output for practical use cases.
 <br>
 
# V. ADDITIONAL MATERIALS
# VI. REFERENCES

<p align="justify">
 
+ [1]S. Cristina, “A Gentle Introduction to OpenCV: An Open Source Library for Computer Vision and Machine Learning - MachineLearningMastery.com,” MachineLearningMastery.com, Oct. 21, 2023. https://machinelearningmastery.com/a-gentle-introduction-to-opencv-an-open-source-library-for-computer-vision-and-machine-learning/?fbclid=IwY2xjawHIA3VleHRuA2FlbQIxMAABHUu1JPVqueoN5f25_iY239JURd2JbGF0wWyt1mVQuEUdfTsE08pqk4mIIA_aem_O46AIkMELobwutrwZ2go0Q <br>
+ [2]Khushi, “Dog Breed Image Dataset,” Kaggle.com, 2024. https://www.kaggle.com/datasets/khushikhushikhushi/dog-breed-image-dataset
‌
‌
