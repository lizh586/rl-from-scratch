import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    """ 
    image:  HxW numpy array
    kernel: KxK numpy array
    返回:   H_out x W_out numpy array    
    """
    H, W = image.shape
    K, _ = kernel.shape
    
    image_pad = np.pad(image, ((padding,padding), (padding, padding)), mode='constant', constant_values=0)
    H_out = int((H+2*padding-K)/stride) + 1
    W_out = int((W+2*padding-K)/stride) + 1
    output = np.zeros((H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):

            h_start = i * stride
            w_start = j * stride
            region = image_pad[h_start:h_start+K,w_start:w_start+K]
            output[i,j] = np.sum(region * kernel)

    return output