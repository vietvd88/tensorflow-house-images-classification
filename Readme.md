In this project, we retrain Inception's Final Layer for new classes in tensorflow framework.  
For more details, please read 2 below articles:  
-   https://www.tensorflow.org/versions/master/how_tos/image_retraining/index.html  
-   https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0

Here are steps to create training data model over 60,000 images into 4 classes:  floor plans,  map, inside, outside.  

1. **In project folder (Ex: **house**), creating training/ for classified images.**
   Create subfolder accordingly:
    + house/
        + training/
            + floor_plans/
            + map/
            + inside/
            + outside/
2.  **Classify first 200 images manually, move images to folder above accordingly.**  
3. **Retrain Inceptionv3 network in tensorflow as steps:**  
    3.1   Run docker container and map project folder into docker container:

        docker run -it -v ~/house:/house  gcr.io/tensorflow/tensorflow:latest-devel`
        cd /house`
    
    3.2 Training images:  

        python tensorflow/examples/image_retraining/retrain.py 
            --bottleneck_dir=/house/bottlenecks \  
            --how_many_training_steps 4000 \  
            --model_dir=/house/inception \  
            --output_graph=/house/retrained_graph.pb \  
            --output_labels=/house/retrained_labels.txt \  
            --image_dir /house/training.

    Training output is a graph and folder data: **retrained_graph.pb, retrained_labels.txt**  
    
    3.3 Classify next images by label_image.py
    
        python label_image.py --output_dir=prediction/ --image_path=unclassified-images/ --threshold=90  
        
    - image_path: a unclassified folder or image
    - output_dir: a folder to hold images after classifying
    - threshold: Percent threshold to decide an image is belong to a class  
    
    After this commands, images will be moved to images folder above accordingly.
    
    3.4 Check again images folder above manually to correct if neccessary.
    
    **_Repeat steps from 3.2 to 3.4 for all images to create better training model_**
   
4. **Use label_image.py script to predict images with training model data above.**  
    Finally, we can use training mode data above to predict images

        python label_image.py --output_dir=prediction/ --image_path=unclassified-images/ --threshold=90  
        
    - image_path: a unclassified folder or image
    - output_dir: a folder to hold images after classifying
    - threshold: Percent threshold to decide an image is belong to a class  
    
    **Example:**
        
        python label_image.py --image_path=tests/house12.jpg
        
    **Output:**
    >   outside (score = 0.99226)
    >   inside (score = 0.00475)
    >   map (score = 0.00227)
    >   floor plans (score = 0.00072)

    With result above, outside probability = 99.22%