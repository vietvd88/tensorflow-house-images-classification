In this project, we retrain Inception's Final Layer for new classes in tensorflow framework.  
For more details, please read 2 below articles:  
    - https://www.tensorflow.org/versions/master/how_tos/image_retraining/index.html  
    - https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0  
Here are steps I used to classify over 60,000 images into 4 classes: floor plans, map, inside, outside.  

### 1. First of all, I select about 200 images for each class by hand.

### 2. After that, I use them to retrain Inceptionv3 network in tensorflow as following:  
    - create a "house/training" folder contains 4 folders corresponding to 4 classes: floor plans, map, inside, outside. Each folder contains 200 images which selected by hand.  
        here is folder structure after creating:  
        + house:
            + training:
                + floor_plans
                + map
                + inside
                + outside
    - create a docker image and map "house" folder with "house" folder in docker by command:  
        docker run -it -v ~/house:/house  gcr.io/tensorflow/tensorflow:latest-devel

    - run "retrain" command in docker image console:  
        python tensorflow/examples/image_retraining/retrain.py \  
            --bottleneck_dir=/house/bottlenecks \  
            --how_many_training_steps 4000 \  
            --model_dir=/house/inception \  
            --output_graph=/house/retrained_graph.pb \  
            --output_labels=/house/retrained_labels.txt \  
            --image_dir /house/training  
    - create python script - lable_image.py - to classify images by using retrained_graph.pb and retrained_labels.txt.

### 3. After having lable_image.py script. I classify about 5000 images from 60,000 images and check manually to quarantee that they are classified correctly. We will have training data set contains some thousands of images after this step.  

### 4. Continue running retrain step to archive a better model. We loop step 2->3 about 3-4 times to have the best model.  

### 5. Use lable_image.py script to predict with all remaining images.  
Here is project folder structure after classification:  
    - bottlenecks : a folder stores cached training data for training process  
    - inception : a folder stores google's inceptionv3 model data  
    - label_image.py: python script is used to classify images in training-images folder.    
        this script will use retrained model retrained_graph.pb and labels etrained_labels.txt to decide the classification for an image.  
    - prediction : python script will detect images and move them to below subfolder of this folder  
        - floor_plans  
        - inside  
        - map  
        - outside  
    - retrained_graph.pb : graph model which was be retrained from google's inceptionv3 model  
    - retrained_labels.txt : contains a list of label we want to classify (floor plans, inside, map, outside)  
    - training : image data is used for training process  
        - floor_plans  
        - inside  
        - map  
        - outside  
    - unclassified-images : contains unclassified images  