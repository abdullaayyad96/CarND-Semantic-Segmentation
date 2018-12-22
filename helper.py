import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
import cv2
import math
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm



class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape, num_classes):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*.png'))}
        background_color = np.array([255, 0, 0])
        my_lane_color = np.array([255, 0, 255])
        other_lane_color = np.array([0, 0, 0])
        colors = np.array([background_color, my_lane_color, other_lane_color])

        if batch_size == -1:
            batch_size = len(image_paths)

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]
                
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
                
                temp_image = np.ndarray((*image_shape, num_classes))
                for i in range(num_classes):
                    gt_bg = np.all(gt_image == colors[i], axis=2)
                    gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                    temp_image[:,:,i] = gt_bg[:,:,0]
                
                gt_image = temp_image
                
                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape, num_classes):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        
        my_lane_color = np.array([[0, 255, 0, 127]])
        other_lane_color = np.array([[0, 0, 255, 127]])
        output_colors = np.array([my_lane_color, other_lane_color])
        
        street_im = scipy.misc.toimage(image)
        for i in range(num_classes-1):
            tem_im_softmax = im_softmax[0][:, (i+1)].reshape(image_shape[0], image_shape[1])
            segmentation = (tem_im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, output_colors[i])
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, num_classes):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape, num_classes)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

def augmentimg(image2aug, gt_image2aug):
    #generate augmented version of images    
    #random integer to select which selector to use
    method_selector = random.randint(1,4)
    
    image_shape = image2aug.shape
    
    if(method_selector == 1):
        #Apply prespective transfromation
        
        #getting random vertices of the image
        x1 = random.randint(0, math.ceil(image_shape[1]/6))
        x2 = random.randint(math.ceil(5 * image_shape[1]/6), image_shape[1])
        y1 = random.randint(0, math.ceil(image_shape[0]/6))
        y2 = random.randint(math.ceil(5 * image_shape[0]/6), image_shape[0])
        
        #applying prespective transformation
        pts1 = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
        pts2 = np.float32([[0,0],[image_shape[1],0],[0,image_shape[0]],[image_shape[1],image_shape[0]]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        output_img = cv2.warpPerspective(image2aug,M,(image_shape[1],image_shape[0]))
        gt_output_img = cv2.warpPerspective(gt_image2aug,M,(image_shape[1],image_shape[0]))
        
    elif(method_selector == 2):
        #apply translation
        
        #getting random shifts for the image
        shift_x = random.randint(-math.ceil(image_shape[1]/7), math.ceil(image_shape[1]/7))
        shift_y = random.randint(-math.ceil(image_shape[0]/7), math.ceil(image_shape[0]/7))
        
        #applying translation
        M = np.float32([[1,0,shift_x],[0,1,shift_y]])
        output_img = cv2.warpAffine(image2aug,M,(image_shape[1],image_shape[0]))
        gt_output_img = cv2.warpAffine(gt_image2aug,M,(image_shape[1],image_shape[0]), borderValue=[255,255,255])
        
    elif(method_selector == 3):
        #apply rotation
        
        #obtaining random angle to rotate
        angle2rotate = random.randint(-7, 7)
        
        #applying rotation
        M = cv2.getRotationMatrix2D((math.ceil(image_shape[1]/2),math.ceil(image_shape[0]/2)),angle2rotate,1)
        output_img = cv2.warpAffine(image2aug,M,(image_shape[1],image_shape[0]))
        gt_output_img = cv2.warpAffine(gt_image2aug,M,(image_shape[1],image_shape[0]), borderValue=[255, 255, 255])
        
    elif(method_selector == 4):
        #change brightness and contrast
        
        alpha = random.uniform(0.7, 1.3)
        beta = random.randint(-20, 20)
        
        output_img = alpha*image2aug + beta
        np.putmask(output_img, output_img>255, 255)
        np.putmask(output_img, output_img<0, 0)
        output_img = np.floor(output_img)
        gt_output_img = gt_image2aug
        
    return output_img, gt_output_img


def apply_augmentation(data_dir, output_dir):
    #load, augment and save images
    image_paths = glob(os.path.join(data_dir, 'data_road/training/image_2', '*.png'))
    label_paths = {}    
    for path in glob(os.path.join(data_dir, 'data_road/training/gt_image_2', '*_road_*.png')):
        label_paths[re.sub(r'_(lane|road)_', '_', os.path.basename(path))] = path
        
    # Make folder for augmented images
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir+'/image_2')
    os.makedirs(output_dir+'/gt_image_2')
    
    #iterate over images
    for i in range(len(image_paths)):
        if 'augmented' not in image_paths[i]:
            image = scipy.misc.imread(image_paths[i])
            gt_image = scipy.misc.imread(label_paths[os.path.basename(image_paths[i])])

            #apply augmentation
            augmented_image, gt_augmented_image = augmentimg(image, gt_image)

            #save original image
            scipy.misc.imsave((output_dir+'/image_2/original_{}.png').format(i), image)
            scipy.misc.imsave((output_dir+'/gt_image_2/original_{}.png').format(i), gt_image)
            #save augmented image
            scipy.misc.imsave((output_dir+'/image_2/augmented_{}.png').format(i), augmented_image)
            scipy.misc.imsave((output_dir+'/gt_image_2/augmented_{}.png').format(i), gt_augmented_image)
               
        