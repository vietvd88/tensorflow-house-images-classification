import tensorflow as tf, sys
from shutil import move
from os import listdir
from os.path import isfile, join, isdir
import imghdr

DEFAULT_THRESHOLD = 90
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('output_dir', '',
                           """Absolute path in order to move classified images into.""")
tf.app.flags.DEFINE_string('image_path', '',
                            """Absolute path to a folder or an image which need to classify""")
tf.app.flags.DEFINE_integer('threshold', DEFAULT_THRESHOLD,
                            """Percent threshold to decide an image is belong to a class""")
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
        
        output_result(image_file, predictions[0], label_lines)

def output_result(image_file, prediction, label_lines):
    # Sort to show labels of first prediction in order of confidence
    top_k = prediction.argsort()[-len(prediction):][::-1]
    
    #print out result to console
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = prediction[node_id]
        print('%s (score = %.5f)' % (human_string, score))
    
    # move image file to right folder
    best_node_id = top_k[0]
    best_score = prediction[best_node_id]
    if (FLAGS.output_dir != '') & (best_score * 100 > FLAGS.threshold):
        target_dir = join(FLAGS.output_dir, label_lines[best_node_id])
        move(image_file, target_dir)

def classify_folder(folder_path, label_lines):
    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    for file in onlyfiles:
        file_path = join(folder_path, file)
        if imghdr.what(file_path) == 'jpeg':
            classify_image(file_path, label_lines)

image_path = (FLAGS.image_path if FLAGS.image_path else
    sys.exit('There are no image path!'))

create_graph()
label_lines = read_labels()
if (isdir(image_path)):
    classify_folder(image_path, label_lines)
else:
    classify_image(image_path, label_lines)



