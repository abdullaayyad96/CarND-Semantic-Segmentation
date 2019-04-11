#!/usr/bin/env python3
import os.path
import tensorflow as tf
import numpy as np
import scipy.misc
import helper
import sys
import os

def run(input_image):  
    #apply NN model on input image and generate output image
      
    num_classes = 3
    model_dir = 'model/model.meta'
    NN_input_shape = (160, 576)
    
    my_lane_color = np.array([[0, 255, 0, 127]])
    other_lane_color = np.array([[0, 0, 255, 127]])
    output_colors = np.array([my_lane_color, other_lane_color])
    
    image = scipy.misc.imresize(scipy.misc.imread(input_image), NN_input_shape)
    
    with tf.Session() as sess:
        
        # load pre trained model 
        saver = tf.train.import_meta_graph(model_dir)
        
        saver.restore(sess, tf.train.latest_checkpoint('model'))
        
        graph = tf.get_default_graph()
       
        logits = graph.get_tensor_by_name('logits:0')    
        keep_prob = graph.get_tensor_by_name('keep_prob:0')        
        input_tensor = graph.get_tensor_by_name('image_input:0')
        
        #run NN model
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, input_tensor: [image]})
        
        #create new picture
        street_im = scipy.misc.toimage(image)
        for i in range(num_classes-1):
            tem_im_softmax = im_softmax[0][:, (i+1)].reshape(NN_input_shape[0], NN_input_shape[1])               
            segmentation = (tem_im_softmax > 0.5).reshape(NN_input_shape[0], NN_input_shape[1], 1)
            mask = np.dot(segmentation, output_colors[i])
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im.paste(mask, box=None, mask=mask)
    
    return street_im


def main(argvs):
    
    input_img = ''
    output_dir = './'
    
    #check arguments
    if (len(argvs) >= 2) and (argvs[1][-3:].upper() == "PNG"):
        input_img = argvs[1]
        
        if (len(argvs) >= 3):
            output_dir = argvs[2]            
            if (output_dir[-1] != "/"):
                output_dir += "/"
                
    else:
        print("Please provide the path to an .png image")
        sys.exit()
    
    #run segmentation
    output_image = run(input_img)
    
    #save output
    output_file = os.path.join(output_dir, os.path.basename(input_img))
    scipy.misc.imsave(output_file, output_image)
                
        
if __name__ == '__main__':
    main(sys.argv)
    sys.exit()
