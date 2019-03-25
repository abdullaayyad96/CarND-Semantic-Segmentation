#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper

def run():    
    num_classes = 3
    data_dir = 'data'
    runs_dir = 'runs'
    model_dir = 'model/model.meta'
    image_shape = (160, 576)
    
    with tf.Session() as sess:
        
        # load pre trained model 
        saver = tf.train.import_meta_graph(model_dir)
        
        saver.restore(sess, tf.train.latest_checkpoint('./'))
            
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, num_classes)
                
        
if __name__ == '__main__':
    run()