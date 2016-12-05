import tensorflow as tf, sys
from shutil import move
from os import listdir
from os.path import isfile, join, isdir
import imghdr

def create_graph():
  # Unpersists graph from file
    with tf.gfile.FastGFile("/house/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def read_labels():
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("/house/retrained_labels.txt")]
    return label_lines

def classify_image(image_file, label_lines):
    print('=============== image_file: ' + image_file + " ===============")
    
    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_file, 'rb').read()

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
            # if (human_string == 'map') & (score * 100 > 80):
            #     move(image_file, '/house/prediction/map/')
            # if (human_string == 'inside') & (score * 100 > 80):
            #     move(image_file, '/house/prediction/inside/')
            # if (human_string == 'outside') & (score * 100 > 80):
            #     move(image_file, '/house/prediction/outside/')
            # if (human_string == 'floor plans') & (score * 100 > 80):
            #     move(image_file, '/house/prediction/floor_plans/')

def classify_folder(folder_path, label_lines):
    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    for file in onlyfiles:
        file_path = join(folder_path, file)
        if imghdr.what(file_path) == 'jpeg':
            classify_image(file_path, label_lines)

image_path = sys.argv[1]
create_graph()
label_lines = read_labels()
if (isdir(image_path)):
    classify_folder(image_path, label_lines)
else:
    classify_image(image_path, label_lines)
