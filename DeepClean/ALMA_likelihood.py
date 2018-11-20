import tensorflow as tf
import numpy as np

class ALMA_sampler(object):
    '''
    This class will perform ALMA-like processing of a given image.  
    It will also serve as our clean modeling utensil, containing the
    likelihood that will be fed to the RIM
    
    some notes:
    
    We give it the true image as a placeholder, along with the sampling UVGRID.  
    An interesting question is whether or not this causes a problem with the optimization.
    Time will tell.
    
    Note:  The tensorflow FFT is wrong by about 0.0001 compared to numpy fft.  Hopefully they fix it
    But don't count on it for now.  It seems to do "ok" at the moment.
    
    The UVGRID fed to the placeholder MUST be the fftshift version.  Tensorflow doesn't have fftshift
    
    The comparison to visibilities is made by scaling the noise by 1/sqrt(Nbl) in each bin.  This is 
    equivalent to averaging each of the visibilities in each bin (rather than summing).  It makes for
    a more natural comparison to the visibility data, but be warned: the resulting beam from the dirty
    images is different from what we usually use.  This may cause errors if we take in the convolved
    images.

    '''
    def __init__(self,img_pl,UVGRID,noise,noise_std,gridded_vis_input=False,full_dim=False):
        '''
        Initialize the object.  Lets have img_pl be the shape we expect to be fed to the network [m,N,N,1]
        and do transposing to reshape things as we need.
        '''

        self.vis_input = gridded_vis_input

        if gridded_vis_input is False:

            N,H,W,C  = img_pl.get_shape().as_list()
            padding = tf.constant([[0,0],[H/2,H/2],[W/2,W/2],[0,0]])

            # to fft in tensorflow, you must have the right type and shape.
            img = tf.cast(img_pl,tf.complex64)
            img = tf.pad(img,padding)
            img = tf.transpose(img,[0,3,1,2])
            
            # Expected noise on an unbinned visibility is the standard deviation of the real noise * sqrt(2)
            self.noise_std = noise_std 
            
            noise = tf.transpose(tf.cast(noise,tf.complex64),[0,3,1,2])
            UVGRID = tf.transpose(UVGRID,[0,3,1,2])

            # normalization factor
            self.N = tf.cast(tf.to_float(tf.reduce_prod(tf.shape(img_pl)[-3:])),dtype=tf.complex64)
            self.N2 = tf.cast(tf.to_float(tf.reduce_prod(tf.shape(noise)[-2:])),dtype=tf.complex64)
        
            # All of the visibilities
            self.vis_full_exact = tf.fft2d(img) / tf.cast(tf.sqrt(self.N),dtype=tf.complex64)
            noise_ft = tf.fft2d(noise) / tf.sqrt(self.N2)
            noise_uv = tf.divide(noise_ft,tf.add(tf.sqrt(tf.cast(UVGRID,tf.complex64)),tf.constant(1e-10,dtype=tf.complex64)))
            self.vis_full_noisy = tf.add(self.vis_full_exact,noise_uv)

            # Lets create a uv mask to avoid divisions by zero.
            self.uvmask  = tf.cast(tf.not_equal(UVGRID,tf.constant(0)),tf.int32)
            vis_bad2,self.noise_ft = tf.dynamic_partition(noise_ft,self.uvmask,num_partitions=2)
            
        else:
            # We're using gridded visibilities, so the sizes will work out a bit differently
            self.vis_full_noisy = tf.transpose(img_pl,[0,3,1,2])
            self.N = tf.cast(192.0,dtype=tf.complex64)
            UVGRID = tf.transpose(UVGRID,[0,3,1,2])
            noise = tf.transpose(tf.cast(noise,tf.complex64),[0,3,1,2])
            self.uvmask  = tf.cast(tf.not_equal(UVGRID,tf.constant(0)),tf.int32)

        # Select only visibilities that have been measured
        vis_bad2,self.vis_noisy= tf.dynamic_partition(self.vis_full_noisy,self.uvmask,num_partitions=2)

        # Select only uv cells that have at least one baseline
        ignored_bl,self.UVGRID = tf.dynamic_partition(UVGRID,self.uvmask,num_partitions=2)
        
        # Scale the noise to correspond with each baseline
        self.sigma = self.noise_std / tf.sqrt(tf.cast(self.UVGRID,tf.float32))
        
        # This is for dirty image computation
        self.UVGRID_full = tf.cast(UVGRID,tf.complex64)

        # If we didn't supply visibilities, we can use this for visualization of noiseless images
        if gridded_vis_input is False:
            self.vis_full_sampled = tf.multiply(self.vis_full_exact,tf.cast(self.uvmask,tf.complex64))

        # Set non-measured baselines to zero for imaging purposes.
        self.vis_full_sampled_noisy = tf.multiply(self.vis_full_noisy,tf.cast(self.uvmask,tf.complex64))

        # Tensor for the dirty images.
        self.DIM = self.get_noisy_dirty_image(return_full=full_dim)

        
    def get_noiseless_dirty_image(self,weighting='natural'):
        '''
        This function produces a noise-free dirty image from simulated data.  It can do both uniform and
        naturally weighted images, and is used mostly for debugging purposes.
        '''
        if weighting == 'uniform':
            return tf.transpose(tf.real(tf.ifft2d(self.vis_full_sampled) * \
                                tf.cast(tf.sqrt(self.N),dtype=tf.complex64)) , \
                                [0,2,3,1])[:,192/2:3*192/2,192/2:3*192/2,:]
        else:
            if self.vis_input:
                dim_full_noiseless = tf.transpose(tf_fftshift(tf.real(tf.ifft2d(tf.multiply(self.vis_full_sampled,\
                                                  tf.cast(self.UVGRID_full,dtype=tf.complex64))) * \
                                                  tf.cast(tf.sqrt(self.N),dtype=tf.complex64))),[0,2,3,1])
            else:
                dim_full_noiseless = tf.transpose(tf.real(tf.ifft2d(tf.multiply(self.vis_full_sampled,\
                                                  tf.cast(self.UVGRID_full,dtype=tf.complex64))) * \
                                                  tf.cast(tf.sqrt(self.N),dtype=tf.complex64)),[0,2,3,1])
            N,H,W,C = dim_full_noiseless.get_shape().as_list()
            return dim_full_noiseless[:,H/2-192/2:H/2+192/2,W/2-192/2:W/2+192/2,:]
        


    def get_noisy_dirty_image(self,weighting='natural',return_full=False):
        if weighting == 'uniform':
            return tf.transpose(tf.real(tf.ifft2d(self.vis_full_sampled_noisy) * \
                                tf.cast(tf.sqrt(self.N),dtype=tf.complex64)) , 
                                [0,2,3,1])[:,192/2:3*192/2,192/2:3*192/2,:]
        else:
            if self.vis_input:
                dim_full = tf.transpose(tf_fftshift(tf.real(tf.ifft2d(tf.multiply(self.vis_full_sampled_noisy,\
                                        tf.cast(self.UVGRID_full,dtype=tf.complex64))) * \
                                        tf.cast(tf.sqrt(self.N),dtype=tf.complex64))),[0,2,3,1])
            else:
                dim_full = tf.transpose(tf.real(tf.ifft2d(tf.multiply(self.vis_full_sampled_noisy,\
                                        tf.cast(self.UVGRID_full,dtype=tf.complex64))) * \
                                        tf.cast(tf.sqrt(self.N),dtype=tf.complex64)),[0,2,3,1])
            
            N,H,W,C  = dim_full.get_shape().as_list()
            
            if return_full == False:
                # return only the center 192 pixels (this is temporary)
                return dim_full[:,H/2-192/2:H/2+192/2,W/2-192/2:W/2+192/2,:]
            else:
                return dim_full
            
        
    def loglikelihood(self,img_pred):
        '''
        This is the log-likelihood of the model.  It is computed via a fourier transform
        of the image, then a selection of the visibilities that were measured, a subtraction
        from the observed signal, and a scaling by the expected noise.
        '''
        
        # This handles the padding dimensions.
        N,C,H,W  = self.vis_full_noisy.get_shape().as_list()
        N2,H2,W2,C2 = img_pred.get_shape().as_list()
        
        # this is the size of the padding
        padding = tf.constant([[0,0],[(H-H2)/2,(H-H2)/2],[(W-W2)/2,(W-W2)/2],[0,0]])
        
        # Pad the visibilities, and transpose for FFT
        img_pred = tf.pad(tf.cast(img_pred,dtype=tf.complex64),padding)
        img_pred = tf.transpose(img_pred,[0,3,1,2])
        
        # If we specified gridded visibilities, then we need to fftshift 
        # because FFTs assume the center is in the upper left corner...
        if self.vis_input:
            img_pred = tf_fftshift(img_pred)

        # FFT the image and normalize
        vis_pred_full = tf.fft2d(img_pred) / tf.cast(tf.sqrt(self.N),dtype=tf.complex64)

        # Dynamic partition to select out the measured baselines
        temp,vis_pred = tf.dynamic_partition(vis_pred_full,self.uvmask,num_partitions=2)

        # Compute the residual error
        difference = tf.subtract(vis_pred,self.vis_noisy)
        
        # Scale by the noise and sum to get the chi-squared
        chi2 = tf.reduce_sum(tf.divide(tf.square(tf.abs(difference)) , tf.square(self.sigma)))
        return chi2

    def predicted_DIM(self,img_pred):
        '''
        A function to predict the dirty image from an image
        '''
        N,C,H,W  = self.vis_full_noisy.get_shape().as_list()
        N2,H2,W2,C2 = img_pred.get_shape().as_list()

        padding = tf.constant([[0,0],[(H-H2)/2,(H-H2)/2],[(W-W2)/2,(W-W2)/2],[0,0]])

        img_pred = tf.pad(tf.cast(img_pred,dtype=tf.complex64),padding)
        img_pred = tf.transpose(img_pred,[0,3,1,2])
        if self.vis_input:
            img_pred = tf_fftshift(img_pred)

        vis_pred_full = tf.fft2d(img_pred) / tf.cast(tf.sqrt(self.N),dtype=tf.complex64)
        dim_full = tf.transpose(tf.real(tf.ifft2d(tf.multiply(vis_pred_full , \
                                tf.cast(self.UVGRID_full,dtype=tf.complex64))) * \
                                tf.cast(tf.sqrt(self.N),dtype=tf.complex64)),[0,2,3,1])

        # Dimensions are the same shape as the input image.
        return dim_full[:,H/2-H2/2:H/2+H2/2,W/2-W2/2:W/2+W2/2,:]



def tf_fftshift(im_pl):
    '''
    performs numpy's fftshift, along the last two axes of a tensor (this way it can be vectorized across batches)
    '''
    left,right = tf.split(im_pl,2,axis=3)
    horizontal_shifted = tf.concat([right,left],axis=3)
    up , down = tf.split(horizontal_shifted,2,axis=2)
    transformed = tf.concat([down,up],axis=2)
    return transformed
