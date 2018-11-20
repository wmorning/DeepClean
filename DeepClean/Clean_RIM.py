


# Normal Python stuff
from time import gmtime, strftime
from os import path
import os
import time
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

# Putzky's RIM libraries.
import iterative_inference_learning.layers.iterative_estimation as iterative_estimation
import iterative_inference_learning.layers.loopfun as loopfun
from iel_experiments.models import decorate_rnn,conv_rnn

# Other DeepClean things we need
import DeepClean as DC
from DeepClean.get_data import read_data_batch


class Clean_RIM(object):
    '''
    An object class to represent the Recurrent Inference Machine of Putzky & Welling (2017).
    Ive made a few updates of my own so that it is more designed for use with ALMA data.
    In particular, the convolutions are much larger, and the likelihood function is properly
    tailored to visibility data.
    '''
    def __init__(self,numpix_side,pix_res,max_noise_rms,visibility_input=False):
        
        # Honestly, we don't need all that much here.
        self.numpix_side = numpix_side
        self.pix_res = pix_res
        self.max_noise_rms = max_noise_rms
        self.visibility_input = visibility_input
        
    def Initialize_placeholders(self):
        '''
        Create the placeholders
        '''
        # syntax simplification
        npx = self.numpix_side
        
        if self.visibility_input:
            # We oversample the gridded visibilities more for the sake of accuracy
            oversample_uv = 16
        else:
            # We always oversample the uv plane slightly to avoid boundary aliasing
            oversample_uv = 2

        # true image (unraveled, then properly shaped)
        self.y_ = tf.placeholder(dtype=tf.float32,shape=[None,npx**2])
        self.y_image = tf.reshape(self.y_,[-1,npx,npx,1])
        
        self.uvgrid = tf.placeholder(dtype=tf.int32,shape=[None,npx*oversample_uv,npx*oversample_uv,1])
        self.noise = tf.placeholder(dtype=tf.complex64,shape=[None,npx*oversample_uv,npx*oversample_uv,1])
        self.noise_rms = tf.placeholder(dtype=tf.float32,shape=[None,1,1,1])
        
        self.x_init = tf.zeros_like(self.y_image)
        
        # needed for batchnorm within the network
        self.is_training = tf.placeholder(tf.bool, [] , name='is_training')
        
    def Configure_Network(self,Ntsteps,Conv_size=11,Nchan_RIM=32,Nchan_img=1,RIM_depth=1,n_pseudo=1,full_dim=False):
        '''
        This will setup the meat and bones of the RIM.  A lot of it we will never touch once its created
        so we won't add it to the object (unless we have to).
        '''
        
        self.likelihood_fn_object = DC.ALMA_sampler(self.y_image,self.uvgrid,self.noise,self.noise_rms,gridded_vis_input=self.visibility_input,full_dim=full_dim)
        
        # Maximum number of iterations
        T = tf.constant(Ntsteps, dtype=tf.int32, name='T')
        
        def error_grad(x_trial):
            return tf.gradients(self.likelihood_fn_object.loglikelihood(self.param2image(x_trial)),x_trial)[0]

        
        # From Putzky & Welling's experiments, define the RIM cell as a stacked convolutional GRU.
        cell , output_function = conv_rnn.gru(Conv_size, [Nchan_RIM]*RIM_depth , is_training = self.is_training)
        
        # All of this stuff was necessary to get the RIM to work.  
        # It lets everything be passed to the decorate rnn function.
        output_shape_dict = {'mu':Nchan_img}
        output_shape_dict.update({'pseudo': Nchan_img*n_pseudo})
        output_transform_dict = {}
        output_transform_dict.update({'all':[tf.identity]})
        output_transform_dict.update({'mu': [error_grad]})
        output_transform_dict.update({'pseudo':[loopfun.ApplySplitFunction(error_grad, 4 - 1, n_pseudo)]})
        
        input_func, output_func, init_func, output_wrapper = decorate_rnn.init(rank=4, output_shape_dict = output_shape_dict,\
                                                                                output_transform_dict = output_transform_dict,\
                                                                                init_name = 'mu', ofunc = output_function ,\
                                                                                accumulate_output=True)
        # We feed a shifted version of the input image
        x_init_feed = self.image2param(tf.maximum(tf.minimum(self.x_init,1.-1e-4),1e-4))
        
        # Finally complete the RIM graph
        alltime_output, final_output, final_state, p_t, T_ = \
            iterative_estimation.function(x_init_feed, cell, input_func, output_func, init_func, T=T)
        
        # Add the outputs to the object
        self.alltime_output = tf.identity(self.param2image(output_wrapper(alltime_output, 'mu', 4)),name='alltime_output')
        self.final_output = tf.identity(self.param2image(output_wrapper(final_output, 'mu')),name='final_output')
        
        ## Define loss functions
        self.loss_full = tf.reduce_sum(tf.reduce_mean(p_t * self.lossfun(self.alltime_output, True), reduction_indices=[1]))
        self.loss = tf.reduce_mean(self.lossfun(self.final_output))
                                                                                                                  
    @staticmethod 
    def image2param(x):
        '''
        Until we get the linear output RIM to work, lets keep the sigmoid on the outputs
        '''
        x_temp = tf.log(x) - tf.log(1-x)
        return x_temp
    
    @staticmethod 
    def param2image(x):
        '''
        Until we get the linear output RIM to work, restrict the outputs to be [0,1]
        '''
        x_temp = tf.nn.sigmoid(x)
        return x_temp
        
    def lossfun(self,x_est,expand_dim=False):
        '''
        Loss Function is just the sum of the squared errors.
        '''
        temp_data = self.y_image
        if expand_dim:
            temp_data = tf.expand_dims(temp_data,0)
        return tf.reduce_sum(0.5 * tf.square(x_est - temp_data) , [-3,-2,-1] )
    
    def Restore_session(self,session,restorefile,variables_to_restore=None):
        '''
        Spawn a tf saver instance, and use it to restore the session from a particular checkpoint file.
        '''
        self.saver = tf.train.Saver(variables_to_restore)
        if restorefile is not None:
            self.saver.restore(session,restorefile)

    def Train(self,Nsteps,learning_rate=1e-6,lrd_timescale=5000,lrd_amplitude=0.96,batch_size=2,save_every=1000000,savefile=None,restorefile=None):
        '''
        Train the network for Nsteps (or until your cluster boots you from it)
        '''
        
        global_step = tf.Variable(0, trainable=False,dtype=tf.int32)
        minimize = tf.contrib.layers.optimize_loss(self.loss_full, global_step, learning_rate, "Adam", clip_gradients=5.0,
                                                   learning_rate_decay_fn=lambda lr,s: tf.train.exponential_decay(lr, s,
                                                   decay_steps=lrd_timescale, decay_rate=lrd_amplitude, staircase=True))
        

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        # Create train and test arrays
        X = np.zeros([batch_size,self.numpix_side**2])
        Y = np.zeros([batch_size,8])
        UVGRID = np.zeros([batch_size,self.numpix_side*2,self.numpix_side*2,1])
        NOISE = np.zeros([batch_size,self.numpix_side*2,self.numpix_side*2,1])
        X_test = np.zeros([batch_size*10,self.numpix_side**2])
        Y_test = np.zeros([batch_size*10,8])
        UVGRID_test = np.zeros([batch_size*10,self.numpix_side*2,self.numpix_side*2,1])
        NOISE_test = np.zeros(UVGRID_test.shape)
        read_data_batch(X_test,UVGRID_test,Y_test,NOISE_test,10000,'test')
        
        # Restore the session if there's a checkpoint
        self.Restore_session(sess,restorefile)

        min_test_cost = np.inf
        
        for i in range(Nsteps):
            read_data_batch(X,UVGRID,Y,NOISE,100000,'train')
            train_cost,_ = sess.run([self.loss,minimize],{self.y_:X,self.uvgrid:UVGRID,self.noise:NOISE, self.is_training:True})
            
            if (i % save_every ==0):
                test_cost = 0.
                for j in range(10):
                    dpm = 2
                    test_cost += sess.run(self.loss, {self.y_:X_test[dpm*j:dpm*(j+1),:],self.uvgrid:UVGRID_test[dpm*j:dpm*(j+1),:],\
                                                self.is_training:False,self.noise:NOISE_test[dpm*j:dpm*(j+1),:]})
                test_cost /= 10.
                if test_cost < min_test_cost:
                    min_test_cost = test_cost
                    if savefile is not None:
                        print "Saving Checkpoint"
                        self.saver.save(sess,savefile)
                        
                print "--------------------------------------------"                
                print "Train step:  ", i
                print "Train cost:  ", train_cost
                print "Test cost:   ", test_cost
                print "--------------------------------------------"
            
    def Predict(self,ims):
        '''
        Make a prediction of the image given a set of observed visibilities.
        '''
        if not self.visibility_input:
            raise Exception("Don't you want to use real visibilities when making predictions on data?")
        return

                        
        
