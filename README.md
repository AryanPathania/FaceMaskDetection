# FaceMaskDetection
Check if a person is wearing a facemask or not using pytorch<br>

In this present COVID-19 scenario our normal life is greatly affected. Now, we can't live our lives as freely as we used to. The first precaution we need to take today is to wear a
facemask to protect ourselves along with the ones near us. <br>
In this project we have use deep learning framework(PyTorch) and various libraries along with OpenCV (for real time detection) to detect if the person is wearing mask or not.<br><br>

### Dataset Used<br>
The dataset which is used to train the model is **Face Mask ~12K Images Dataset** from Kaggle. <br>
Originally the dataset set was divided in 3 parts - Training, Validation and Test. I combined training and validation set to increase the training examples and used test set as 
validation set to check for accuracy and other important things.<br>
The data was labeled as **"WithMask, WithoutMask"**<br>
This is the link for the [dataset](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset)<br><br>

### What I did !!!<br>

- First I used a pre trained model i.e. **MobileNetV2** to train the model according to my dataset.
- After saving the model I used OpenCV to allow the program to access my webcam and use the MTCNN model to check if there is face in front of the camera.
- At last I used the frame from the camera and used the model (trained) to check if the person is wearing the mask or not.
<br><br>
### Result<br>
![FaceMaskExample](https://user-images.githubusercontent.com/50714723/103445929-a5cb3080-4c9f-11eb-9974-e86e1c244ec2.gif)
