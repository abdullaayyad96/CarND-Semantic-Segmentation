import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from tensorflow.python.platform import gfile

num_classes = 3
image_shape = (160, 576)
data_dir = '/data'
runs_dir = './runs'
model_name = "/model/frozen.pb"
tests.test_for_kitti_dataset(data_dir)

input_tensor_name = 'image_input:0'
keep_prob_tensor_name = 'keep_prob:0'
logits_tensor_name = 'logits:0'


def load_graph(graph_file, use_xla=False):
    jit_level = 0
    config = tf.ConfigProto()
    if use_xla:
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level
    
    config.gpu_options.allow_growth=True
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        ops = sess.graph.get_operations()
        n_ops = len(ops)
        return sess, sess.graph, ops
'''
sess, graph, base_ops = load_graph(model_name)

input_image = sess.graph.get_tensor_by_name(input_tensor_name)
keep_prob = sess.graph.get_tensor_by_name(keep_prob_tensor_name)
logits = sess.graph.get_tensor_by_name(logits_tensor_name)

helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
'''    
with tf.Session() as sess:
    '''
    with gfile.FastGFile(model_name,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    '''
    new_saver = tf.train.import_meta_graph('/model/model_3.meta')
    new_saver.restore(sess, '/model/model_3')
    input_image = sess.graph.get_tensor_by_name(input_tensor_name)
    keep_prob = sess.graph.get_tensor_by_name(keep_prob_tensor_name)
    logits = sess.graph.get_tensor_by_name(logits_tensor_name)

    helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, 3)
