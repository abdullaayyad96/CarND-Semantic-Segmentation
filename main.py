#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
        
    #load tensors
    input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return input_tensor, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
  
    #1x1 convolution to obtain desired number of num_classes
    conv1x1_1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, (1,1), padding='same',
									kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer7_1x1conv')
   
    #deconvolution 
    deconv_1 = tf.layers.conv2d_transpose(conv1x1_1, num_classes, 4, 2, padding='same', 
											kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer7_deconv')
    
    #scale layer4 
    layer4_scaled = tf.scalar_mul(0.01, vgg_layer4_out)
    
    #1x1 convolution for scaled layer 4
    layer4_conv1x1 = tf.layers.conv2d(layer4_scaled, num_classes, 1, (1,1), padding='same',
										kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer4_1x1conv')
    
    #first skipping layer
    skip_1 = tf.add(deconv_1, layer4_conv1x1, name='layer_7_add_4')

    #deconvolution 
    deconv_2 = tf.layers.conv2d_transpose(skip_1, num_classes, 4, 2, padding='same',
											kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.contrib.layers.xavier_initializer() name='layer_74_deconv')
    
    #scale layers3 
    layer3_scaled = tf.scalar_mul(0.0001, vgg_layer3_out)
    
    #1x1 convolution for scaled layer 3
    layer3_conv1x1 = tf.layers.conv2d(layer3_scaled, num_classes, 1, (1,1), padding='same',
										kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer3_1x1conv')
    
    #skipping layer 2
    skip_2 = tf.add(deconv_2, layer3_conv1x1, name='layer_7_add_3')

    #output layer
    output_layer = tf.layers.conv2d_transpose(skip_2, num_classes, 16, 8, padding='same',
												kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='output_layer')
        
    return output_layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    #reshape logits to be same size of labels
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) #regularization loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn_last_layer, labels=correct_label), name='cross_entropy_loss') #cross entropy
    #overall loss combining cross entropy and regularization
    overall_loss = tf.add(1.0*sum(reg_loss), cross_entropy_loss, name='total_loss')
  
    #obtain training operation
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    gvs = optimizer.compute_gradients(overall_loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    training_operation = optimizer.apply_gradients(capped_gvs)
    #training_operation = optimizer.minimize(overall_loss, name='train_op')

    return logits, training_operation, overall_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print    out the loss during training.
  param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    #initialize variables
    sess.run(tf.global_variables_initializer())
    
    print("Training...")
    print()

    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        for image, label in get_batches_fn(batch_size):

            optimizer, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5})
            print("Loss: = {:.3f}".format(loss))
            print()
tests.test_train_nn(train_nn)


def run():
    num_classes = 3
    image_shape = (160, 576)
    data_dir = 'data'
    augmented_data_dir = 'modified_data'
    runs_dir = 'runs'
    model_dir = 'model'
    tests.test_for_kitti_dataset(data_dir)

    epochs = 20
    batch_size = 5

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    #augment the data
    print('Augmentation starting ...')
    helper.apply_augmentation(data_dir, augmented_data_dir)
    print('Augmentation finished')
    
    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
    learning_rate = 0.0001

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # get batches
        get_batches_fn = helper.gen_batch_function(augmented_data_dir, image_shape, num_classes)

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        
        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)
                
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, num_classes)
        
        # save trained model and frozen graph
        saver = tf.train.Saver()
        tf.train.write_graph(sess.graph_def, model_dir, 'train.pb', as_text=False)
        saver.save(sess, model_dir+'/model')
        output_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['logits'])
        tf.train.write_graph(output_graph, model_dir, 'frozen_graph.pb', as_text=False)
        

if __name__ == '__main__':
    run()
