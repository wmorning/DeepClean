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
    def __init__(self,img_pl,UVGRID,noise,gridded_vis_input=False):
        '''
        Initialize the object.  Lets have img_pl be the shape we expect to be fed to the network [m,N,N,1]
        and do transposing to reshape things as we need.  Note that if gridded visibilities are used img_pl
        should be the same shape as UVGRID and noise (both of which are also placeholders).

        The noise placeholder is used so that noise can be added to data during training, if visibilities are
        used, noise will not ever enter into the observed images or anything (so it can safely be ignored).

        A future version of this will explicitly calculate the chi2 of the observed visibilities given the
        image.  What we have right now is proportional to this (given that all visibilities have the same
        noise), but it is not scaled properly.  For the best possible work, this should be fixed.  
        
        ------------------------------------------------------------
        Inputs:

        img_pl:    A placeholder either for the true image (for training), or for the observed visibilities
                   (when using the network as a prediction machine).

        UVGRID:    A placeholder for the uv sampling (in grid form)  for training, it should be 2x as large
                   as the img_pl, and for prediction, it should be the same size.
        
        noise:     A placeholder for the noise to be added during training.  It should be the same shape as 
                   the uv grid (note that it will not be used given visibility data).

        gridded_vis_input:    A boolean declaring if the input img_pl is gridded visibilities or an image.

        '''

        self.vis_input = gridded_vis_input

        if gridded_vis_input is False:  # data is an image, so we'll have to FT to get visibilities.
            
            # padding the image helps us get a more accurate approximation to the beam.
            N,H,W,C  = img_pl.get_shape().as_list()
            padding = tf.constant([[0,0],[H/2,H/2],[W/2,W/2],[0,0]])

            # to fft in tensorflow, you must have the right type and shape.
            img = tf.cast(img_pl,tf.complex64)
            img = tf.pad(img,padding)
            img = tf.transpose(img,[0,3,1,2])
            noise = tf.transpose(tf.cast(noise,tf.complex64),[0,3,1,2])
            UVGRID = tf.transpose(UVGRID,[0,3,1,2])

            # normalization factor for FFT
            self.N = tf.cast(tf.to_float(tf.reduce_prod(tf.shape(img_pl)[-2:])),dtype=tf.complex64)
        
            # FT to get all possible visibilities
            self.vis_full_exact = tf.fft2d(img) / tf.cast(tf.sqrt(self.N),dtype=tf.complex64)
            
            # noise was given in real space, lets move it to Fourier space as well.
            noise_ft = tf.fft2d(noise) / tf.sqrt(self.N)
            
            # Scale the noise by 1/sqrt of the number of baselines in each cell. Avoid div by 0 for unsampled cells
            noise_uv = tf.divide(noise_ft,tf.add(tf.sqrt(tf.cast(UVGRID,tf.complex64)),tf.constant(1e-10,dtype=tf.complex64)))
            
            # All possible noisy visibilities (including ones that will not be sampled)
            self.vis_full_noisy = tf.add(self.vis_full_exact,noise_uv)
        
        else: # visibilities were provided, so comparatively less processing must be done.

            # the shape has to change to align with the forward model
            self.vis_full_noisy = tf.transpose(img_pl,[0,3,1,2])
            
            # For now, we just need an N, so lets use the pixscale we trained on...
            self.N = tf.cast(192.0,dtype=tf.complex64)
            
            # switching dims for compatibility... again.
            UVGRID = tf.transpose(UVGRID,[0,3,1,2])

            # Is this even necessary?
            noise = tf.transpose(tf.cast(noise,tf.complex64),[0,3,1,2])

        # We can use a mask to avoid divisions by zero later on.  Also lets mask everything we can.
        self.uvmask  = tf.cast(tf.not_equal(UVGRID,tf.constant(0)),tf.int32)
        vis_bad2,self.vis_noisy= tf.dynamic_partition(self.vis_full_noisy,self.uvmask,num_partitions=2)
        ignored_bl,self.UVGRID = tf.dynamic_partition(UVGRID,self.uvmask,num_partitions=2)
        
        # for imaging purposes
        self.UVGRID_full = tf.cast(UVGRID,tf.complex64)
        self.vis_full_sampled_noisy = tf.multiply(self.vis_full_noisy,tf.cast(self.uvmask,tf.complex64))

        # define the dirty image as part of the graph.
        self.DIM = self.get_noisy_dirty_image()
        

    def get_noiseless_dirty_image(self):
        # I'm not going to really do anything with this, but its here.
        return tf.transpose(tf.real(tf.ifft2d(self.vis_full_sampled) * tf.cast(tf.sqrt(self.N),dtype=tf.complex64)),[0,2,3,1])[:,192/2:3*192/2,192/2:3*192/2,:]      

    def get_noisy_dirty_image(self,weighting='natural'):
        # the output image varies depending on how it is weighted.
        if weighting == 'uniform':
            return tf.transpose(tf.real(tf.ifft2d(self.vis_full_sampled_noisy)*tf.cast(tf.sqrt(self.N),dtype=tf.complex64)),[0,2,3,1])[:,192/2:3*192/2,192/2:3*192/2,:]
        else:
            if self.vis_input:
                dim_full = tf.transpose(tf_fftshift(tf.real(tf.ifft2d(tf.multiply(self.vis_full_sampled_noisy,tf.cast(self.UVGRID_full,dtype=tf.complex64)))*tf.cast(tf.sqrt(self.N),dtype=tf.complex64))),[0,2,3,1])
            else:
                dim_full = tf.transpose(tf.real(tf.ifft2d(tf.multiply(self.vis_full_sampled_noisy,tf.cast(self.UVGRID_full,dtype=tf.complex64)))*tf.cast(tf.sqrt(self.N),dtype=tf.complex64)),[0,2,3,1])
            N,H,W,C  = dim_full.get_shape().as_list()
            return dim_full[:,H/2-192/2:H/2+192/2,W/2-192/2:W/2+192/2,:]
            
        
    def loglikelihood(self,img_pred):
        '''
        Given a set of visibilities (either given via an image plus uv sampling or by observed visibilities),
        calculate the loglikelihood (chi2) of an input image.

        Right now, this is proportional to the chi2, but it is not strictly correct.  This is mainly
        due to the difficulty in interpreting the input noise correctly (because of the FFT).  For
        real visibility data, we know the expected noise, so this would be trivial.

        Either way, for the RIM, this should be used in the gradient function (specifically, with tf.gradients).
        
        '''
        # get sizes, and pad/reshape  accordingly
        N,C,H,W  = self.vis_full_noisy.get_shape().as_list()
        N2,H2,W2,C2 = img_pred.get_shape().as_list()            
        padding = tf.constant([[0,0],[(H-H2)/2,(H-H2)/2],[(W-W2)/2,(W-W2)/2],[0,0]])        
        img_pred = tf.pad(tf.cast(img_pred,dtype=tf.complex64),padding)
        img_pred = tf.transpose(img_pred,[0,3,1,2])

        if self.vis_input:
            # the center of the image is our origin, therefore use an fftshift to enforce this with real data.
            img_pred = tf_fftshift(img_pred)
            
        # Forward model the visibilities
        vis_pred_full = tf.fft2d(img_pred) / tf.cast(tf.sqrt(self.N),dtype=tf.complex64)
        temp,vis_pred = tf.dynamic_partition(vis_pred_full,self.uvmask,num_partitions=2)
        
        # visibility residuals
        difference = tf.subtract(vis_pred,self.vis_noisy)
        
        # chi2 is proportional to the squared residuals over the squared noise relative scale
        chi2 = tf.reduce_sum(tf.multiply(tf.square(tf.abs(difference)) , tf.cast(self.UVGRID,tf.float32)))
        return chi2



def tf_fftshift(im_pl):
    '''
    performs numpy's fftshift, along the last two axes of a tensor (this way it can be vectorized across batches)
    '''
    left,right = tf.split(im_pl,2,axis=3)
    horizontal_shifted = tf.concat([right,left],axis=3)
    up , down = tf.split(horizontal_shifted,2,axis=2)
    transformed = tf.concat([down,up],axis=2)
    return transformed
