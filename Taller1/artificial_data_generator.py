import tensorflow as tf
import read_images
from IPython.display import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
import utils
from skimage.transform import resize;
import numpy as np;
from skimage.external import tifffile;
from skimage.transform import resize;
import os;
import shutil;
import read_images;
from ImageAugmenter import ImageAugmenter
from scipy import misc
from PIL import Image
import random
import os;

def reshape_image_to_vect(image_array):
    image_array = image_array.reshape(len(image_array),-1);
    return image_array;

def reshape_image_to_conv(image_array,rows,cols,channels):
    image_array = image_array.reshape((-1,rows,cols,channels)).astype(np.float32)
    return image_array

def numeric_labels_to_one_hot_encodings(binary_labels,number_of_labels):
     binary_labels = (np.arange(number_of_labels) == binary_labels[:,None]).astype(np.float32)
     return binary_labels;

def image_to_numpy_matrix(image):
    width,height = image.size;
    numpy_array = np.array(list(image.getdata()));
    numpy_matrix  = np.reshape(numpy_array,(height,width)).astype(np.uint8);
    return numpy_matrix;

def create_artificial_image_names(image,artificial_images_number):
    artificial_names = [];
    for i in range(artificial_images_number):
        artificial_name = "artificial"+str(i+1)+"_"+image;
        artificial_names.append(artificial_name);
    
    return artificial_names;

def gen_augmented_artificial_images(image,mask,number_of_artificial_images,scale_to_percent,scale_axis_equally,rotation_deg,shear_deg,translation_x_px,translation_y_px):
    aux_image = Image.fromarray(image);
    aux_mask = Image.fromarray(mask);
    width = aux_image.width;
    height = aux_image.height;
    augmented_images = np.ndarray(shape=(number_of_artificial_images,height,width),dtype = np.uint8);
    augmented_masks = np.ndarray(shape=(number_of_artificial_images,height,width),dtype = np.uint8);
    
    #create  fliped images
    if number_of_artificial_images > 0:
        image_horizontal_flip = aux_image.transpose(Image.FLIP_LEFT_RIGHT);
        mask_horizontal_flip = aux_mask.transpose(Image.FLIP_LEFT_RIGHT);
        augmented_images[0] =image_to_numpy_matrix(image_horizontal_flip);
        augmented_masks[0] = image_to_numpy_matrix(mask_horizontal_flip);
        
    if number_of_artificial_images> 1:    
        image_vertical_flip = aux_image.transpose(Image.FLIP_TOP_BOTTOM);
        mask_vertical_flip = aux_mask.transpose(Image.FLIP_TOP_BOTTOM);
        augmented_images[1] =image_to_numpy_matrix(image_vertical_flip);
        augmented_masks[1] = image_to_numpy_matrix(mask_vertical_flip);
    
    #create random augmentations
    if number_of_artificial_images> 2:
        augmenter = ImageAugmenter(width, height, 
                           scale_to_percent=scale_to_percent,
                           scale_axis_equally=scale_axis_equally,
                           rotation_deg=rotation_deg,    
                           shear_deg=shear_deg,       
                           translation_x_px=translation_x_px, 
                           translation_y_px=translation_y_px,  
                           
                           );
    
        for i in range(number_of_artificial_images-2):
            seed = random.randint(0,4294967294)
            augmented_image = augmenter.augment_batch(np.array( [image],dtype=np.uint8),seed = seed);
            augmented_mask = augmenter.augment_batch(np.array( [mask],dtype=np.uint8),seed = seed);
            augmented_images[i+2] = np.rint( augmented_image*255 ).astype(np.uint8);
            augmented_masks[i+2] = np.rint(augmented_mask*255).astype(np.uint8);
    
    
    

    return  augmented_images,augmented_masks;