# PROJECT-3-UEMK-MASK-DETECTION-SYSTEM :star_struck: :computer: :open_file_folder: :pencil2: :notebook:

[![Generic badge](https://img.shields.io/badge/advance-Python3-yellowgreen)](https://shields.io/) [![Generic badge](https://img.shields.io/badge/deep-learning-red)](https://shields.io/) [![Generic badge](https://img.shields.io/badge/artificial-neural%20retwork-yellow)](https://shields.io/) [![Generic badge](https://img.shields.io/badge/convolutional%20-neural%20network-orange)](https://shields.io/) [![Generic badge](https://img.shields.io/badge/MNIST-dataset-green)](https://shields.io/) [![Generic badge](https://img.shields.io/badge/Graphical%20-UI-brightgreen)](https://shields.io/) [![Generic badge](https://img.shields.io/badge/Tesseract-OCR-blue)](https://shields.io/) [![Generic badge](https://img.shields.io/badge/python-3.6-green)](https://shields.io/) 

***This new Mask Detection System using Python is created by Biswarup Bhattacharjee, student of BTECH, in University of Engineering and Management, Kolkata.***

**Email Id: bbiswa471@gmail.com.** 

**Contact No: 916290272740.** 


<p align="left">
<a href="https://www.facebook.com/profile.php?id=100070395300810" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/facebook.svg" alt="biswa2210" height="30" width="40" /></a>
<a href="https://instagram.com/biswarup2210" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/instagram.svg" alt="" height="30" width="40" /></a>
<a href="https://github.com/biswa2210/biswa2210" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/github.svg" alt="" height="30" width="40" /></a>
</p>

## About :point_down: 

<div align="justified">
 
This is a ‘Mask Detection System’ based on real world AI and ML applications. The software detects if people are wearing masks or not. It also has a global notification section of covid cases throughout the world. This software is made with Python. Some modules are Tkinter, Pillow, Beautifulsoup, Plyer, Opencv, Pandas, Requests, Itertools etc. A dataset has been used for training and testing data as images of people wearing masks and without masks with mask. We have selected this idea to make project remembering the present covid situation, importance of masks and also the uprising trend of using AI and ML in many real-world practices. Our project ‘MASK DETECTION SYSTEM’ along with ‘COVID UPDATES’ works based on machine learning concepts. There is a GUI canvas which contains background image changing with time. There are three buttons for different functions. The title of the first button is ‘LIVE DETECTION’. Opening it we get a computer vision using web camera and a bounding rect locating the area. Coming in front of webcam we can see the result. The system detects if we are wearing masks or not. It shows accuracy percentage as per prediction and also the number of people before webcam. The system has been cross checked many times with similar things with mask but in each case, we get 99% correct result. To exit the live detection function we have to press ‘q’ on our keyboard. The next button in the main GUI is ‘COVID UPDATES’. By clicking on this we get a window where we have to write country name and type of covid case. We can select type of cases related Covid-19 like total deaths, total cases, new cases, new deaths, total recovered, all cases. Then we can to click on the show notification button to get a notification pop up in pc. We can also download json or csv format files as record. We have fetched the global updates from worldometer website. The results are updated and correct. The third button on the main GUI is EXIT which is used to exit the program.
   
</div>

**:point_right: [click here to read Project1 Research Paper]()<br>
:point_right: [click here to see Project1 PowerPoint Presentation]()<br>
:point_right: [click here to view or download Project1 Demo Video]()**

<figure class="video_container">
  <video controls="true" allowfullscreen="true" poster="pics/hr1.PNG">
    <source src="SAMPLE OUTPUT/3SEM PROJECT.mp4" type="video/mp4">
  </video>
</figure>


## Purpose :point_down:

<div align="justified">
 
The main goal of our project is to implement ‘MASK DETECTION SYSTEM’ and ‘COVID UPDATES’ using deep learning concept. For making it user friendly our target is to build a ‘GUI’ with buttons integrated on it for using the system. For the live detection idea, we have to detect faces and also detect the accuracy or wearing or not wearing mask. Based on that we have to produce a decision as the output. We also have to work on image data for training model. For Covid Updates section we have to make a GUI and pop notification as per user’s chosen country and type of covid case. We have to also keep an option to download databases of covid cases. To make correct result and get correct and updated data is also our concern.
 
</div>

## Applications :point_down:

1.	In the ‘Live Detection’ part we can place our face before web-cam of our pc and then it will detect that we are wearing mask or not. The number of people will be shown. More than one person can be detected.
2.	The prediction accuracy of with mask or without mask will be shown. We can clearly get an idea that the person is wearing mask or not. So, in this global Covid-19 pandemic situation this system can be used for safety measures.
3.	This system can be used in offices, schools, colleges, railway station, airport, hospitals, shopping malls and any other places to check that people are wearing masks or not.
4.	In the ‘Global Updates’ part we have to write the name of the country. We can select type of cases related Covid-19 like total deaths, total cases, new cases, new deaths, total recovered, all cases. Then we have to click on the show notification part to see the result. We can also download json or csv format files as record. The records fetched from the worldometer website is accurate.
5.	The notification system can be also used anywhere for database analysis of Covid Cases. Students, researchers, officers or anyone can use this for database and research purpose. This will also help for statistical analysis of records.

## Future Scopes :point_down:

1.	This ‘Face Mask Detection’ system can be deployed in any public place or densely populated areas in checking area before entry for Covid-19 safety measures. It will increase awareness of the global pandemic situation and everyone will know the importance of wearing mask.
2.	This system will make people follow the guidelines related Covid-19 and it will also make the situation better by decreasing the number of affected people.
3.	Many important exams which are getting postponed affecting students’ careers. In future using this system many exams and other important tasks can be conducted avoiding any losses.
4.	The global notification idea can be added in other systems and shown in public places through tv or computer to keep people updated and make them alert for their health.
5.	The development ideas, AIML concepts and other knowledges, the global notification or database part will encourage students or researchers to increase their knowledges.


## Folder Structure :point_down:

```bash
PROJECT-1-UEMK
     ├── .idea
     |      ├── inspectionProfiles
     |      |          ├── Project_Default.xml
     |      |          └── profiles_settings.xml
 
```                       

## Making :point_down:

<div align="justified">

Dataset-->We have downloaded external dataset of people’s images with and without masks from Kaggle. The dataset consists 1915 images of people wearing masks and 1918 images of people without wearing masks. In this dataset we have used 80% images as training data and 20% images as testing data.<br>
Model Training--> We have created a training.py file for model training. Here we have set learning rate= 1e-4, epoch= 20 and batch size= 32 for our ease in implementation. Here are two categories of data with mask and without mask. For choosing path we have used os.listdir(path). Then we have converted image to array. We have set the size of the images as (224,224). We have used labelbinarizer for labeling and then taken the data as numpy array. Here we have splitted data in trainX, trainY, testX, testY. We have constructed the training image generator for data augmentation. We have used MobileNetV2 network for working with baseModel and headModel. Here we have kept the head of the model that will be placed on top of the base model. Here we have used AveragePooling2D, Flatten, Dense. We have used activation function relu for solving vanishing gradient problem. We have used dropout 0.5 for avoiding overfitting. After all these implementations we have compiled the model and made it to predict based on data. We have saved the model as mask_detector.model. After that we have plotted a hysteresis graph of epoch (X axis) vs loss-accuracy (Y axis) using python matplotlib library. Then we have saved the image as plot.png.<br>
Detection--> We have created detection.py file for detection after prediction. We have downloaded a dataset and fetched it through facenet and we have fetched our trained model through masknet. Then we initialized our list of faces, their corresponding locations and the list of predictions from our face mask network using three lists faces, locs, preds. For live detection we have used opencv cv2 in python. Here we have set a frame of detection through live camera streaming. Here we have used an infinite loop for keeping the camera on. Here we have made a bounding rect where users have to place their faces. Here we have measured confidence. We have made   livedetect function for live detecting faces. Then we have filtered out weak detections by ensuring the confidence is greater than the minimum confidence If it comes greater than 50% then user is wearing mask. The accuracy percentage is also calculated. Generally, the images are in BGR format of live detection. We made them in RGB format for further implementation. Then after recognition we convert them in grey scale images and also resize them in (224,224). We have kept the sizes of dataset’s images and recognized images same for perfectly comparing them. For recognizing faces a model is also created. The no. of people is also calculated. Our model only makes a prediction if the number of people is greater than 0 or any people is recognized. For live detection we have used VideoStream. Using puttext function we have printed the output, number of people on the screen. After all these we have made a checking if q is pressed then the output screen will be closed using destroyAllWindows function and also stopping the livestreaming.<br>
Covid Updates--> For Covid updates we have created CovidUpdates.py. Here we have used pandas, plyer, beautifulsoup, requests python modules. We have created notifyme function here we have called the notification. We have created list of all cases in header'countries','total_cases','new_cases','total_deaths','new_deaths','total_recovered'. We made getdata function for fetching data. Here we have used beautifulsoup module. We made a section that requests data from html.parser. We have fetched ‘tbody’ class from the code of the worldometer website. Here we have fetched td, tr accordingly. For updated data we have implemented it. We have used append, replace and strip here. We have made a checking that if the input is invalid then for all the cases it will show ‘Not Found’. We have made downdatascsv function for getting data from tbody class and downloading it in .csv format. Here we have fetched all td as ttt and made a loop. We have used downdatasjson function for downloading it in .json format. We have used pandas dataframe for downloading the data in JSON (Javascript Object Notation) and CSV (Comma Separated Values).  Then we have converted the dataframe to json or csv. Pop Notification has been created using plyer module.<br>
Corona Updates GUI--> For the covid updates GUI we have created CoronaGlobalUpdatesGUI.py. We have imported notifyMe, getData, downdatascsv, downdatasjson from CovidUpdates.py. We have created globalUpdates function. We have set the size as 1030x300 and the co-ordinates means where the window will open in pc screen 200+80. We have used root.iconbitmap for two randomly interchanging icons. We made an IntroLabel.  And placed it in (x=0,y=0). We have made EntryLabel and FormatLabel also. We made the background black. We have used ImageLabel  class for GIF which seems to be a moving corona virus. We have used font size, style and color. We have placed the moving corona virus GIF using rigthliveCovid and leftliveCovid. After we have created drop down menu for choosing types of cases. Then we have converted types_of_cases_data to StringVar. We have used drop.place for setting the co-ordinates in (x=786,y=70). Then we have created shownoti function. Here we have fetched the data from https://www.worldometers.info/coronavirus/ and got the notification using checking. Here we have created createCSV and createJSON function and called them in individual buttons and created a notification to show successfully downloaded. We have placed InJson button in x=470,y=150 in the GUI and designed the button. Then we have placed InCsv button in x=620,y=150 and designed the button in GUI. We have placed the submit button in x=320,y=250 and designed it for getting notification. <br>
Main GUI-->For the main canvas window GUI we have created mask_detection_gui.py. We have kept the sounds for clicking in a list named poklist and the background images in pokulist. We For playing sound we have used lambda function. We have created the output window as well as the main GUI using root.geometry("675x500+120+120"). Here  675x500 is the size and 120 is the coordinates. For randomly changing icons we have used root.iconbitmap. Here we have created the three buttons. Here we have designed them with padx, pady, raised, background, font, font colors etc. In the first button we have called livedetect() function. We have called globalUpdates() function in the second button. In the third button we have called root.quit() function to ext the program. Here using canvas.create_window we have opened the functions of live detection and covid updates using buttons.<br>

</div>

## Screenshots :point_down: 

<div align="center">
<a href="pics/hr1.PNG"><img src="pics/hr1.PNG" width="400" height= "300"></a> <a href="pics/hr2.PNG"><img src="pics/hr2.PNG" width="400" height= "300"></a>


</div>

## Block Diagram of Project :point_down:

<div align="center">
<a href="pics/bl.PNG"><img src="pics/bl.PNG" width="800" height= "350"></a>
</div>


