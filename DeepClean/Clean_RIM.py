


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
from iel_experiments.models import superres_rnn, decorate_rnn,conv_rnn

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
    def __init__(self,numpix_side,pix_res,max_noise_rms):
        
        # Honestly, we don't need all that much here.
        self.numpix_side = numpix_side
        self.pix_res = pix_res
        self.max_noise_rms = max_noise_rms
        
    def Initialize_placeholders(self):
        '''
        Create the placeholders
        '''
        # syntax simplification
        npx = self.numpix_side
        
        # true image (unraveled, then properly shaped)
        self.y_ = tf.placeholder(dtype=tf.float32,shape=[None,npx**2])
        self.y_image = tf.reshape(self.y_,[-1,npx,npx,1])
        
        self.uvgrid = tf.placeholder(dtype=tf.int32,shape=[None,npx*2,npx*2,1])
        self.noise = tf.placeholder(dtype=tf.complex64,shape=[None,npx*2,npx*2,1])
        
        self.x_init = tf.zeros_like(self.y_image)
        
        # needed for batchnorm within the network
        self.is_training = tf.placeholder(tf.bool, [] , name='is_training')
        
    def Configure_Network(self,Ntsteps,Conv_size=11,Nchan_RIM=32,Nchan_img=1,RIM_depth=1,n_pseudo=1):
        '''
        This will setup the meat and bones of the RIM.  A lot of it we will never touch once its created
        so we won't add it to the object (unless we have to).
        '''
        
        self.likelihood_fn_object = DC.ALMA_sampler(self.y_image,self.uvgrid,self.noise)
        
        # Maximum number of iterations
        T = tf.constant(Ntsteps, dtype=tf.int32, name='T')
        
        def error_grad(x_trial):
            return tf.map_fn(lambda xin: tf.image.per_image_standardization(xin),\
                                    tf.gradients(self.likelihood_fn_object.loglikelihood(x_trial),x_trial)[0])
        
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
    
    def Restore_session(self,session,restorefile):
        '''
        Spawn a tf saver instance, and use it to restore the session from a particular checkpoint file.
        '''
        self.saver = tf.train.Saver()
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
        return

                        
        
#def log10(x):
#        return tf.log(x)/tf.log(tf.constant(10, dtype=tf.float32))

#def get_psnr(x_est, x_true):
#    rmse = tf.sqrt(tf.reduce_mean(tf.square(x_true - x_est),[-3,-2,-1]))
#    psnr = 20. * (- log10(rmse))

#    return psnr

#def train():
#    model_name = os.environ['WORK']+'/trained_weights/CLEAN_RIM/model_new_likelihood3.ckpt'
#    full_path = model_name

    # DEFINE WARREN's stuff
#    numpix_side = 192
#    batch_size = 2
#    pix_res = 0.04
#    L_side = numpix_side*pix_res
#    global max_noise_rms, max_psf_rms , max_cr_intensity
#    max_trainoise_rms = 0.1
#    max_testnoise_rms = 0.1
#    max_noise_rms = max_testnoise_rms
#    cycle_batch_size = 10
#    num_test_samples = 500
#    global arcs_data_path_1, arcs_data_path_2 , test_data_path_1 , test_data_path_2 , CRay_data_path
#    global lens_data_path_1, lens_data_path_2, testlens_data_path_1, testlens_data_path_2
#    global min_unmasked_flux
#    min_unmasked_flux = 0.75
#    global num_data_dirs
#    num_data_dirs = 2
#    num_training_samples = 100000
#    max_num_test_samples = 1000
#     arcs_data_path_1 = os.environ['WORK'] + '/NAZGUL/ARCS_1/'
#     arcs_data_path_2 = os.environ['WORK'] + '/NAZGUL/ARCS_2/'
#     test_data_path_1 = os.environ['WORK'] + '/NAZGUL/ARCS_1/'
#     test_data_path_2 = os.environ['WORK'] + '/NAZGUL/ARCS_2/'
#
#     min_test_cost = 12.
#
#     execfile('get_data.py')
#
#     X = np.zeros([batch_size,numpix_side**2])
#     Y = np.zeros([batch_size,8])
#     UVGRID = np.zeros([batch_size,numpix_side*2,numpix_side*2,1])
#     NOISE = np.zeros([batch_size,numpix_side*2,numpix_side*2,1])
#     X_test = np.zeros([batch_size*10,numpix_side**2])
#     Y_test = np.zeros([batch_size*10,8])
#     UVGRID_test = np.zeros([batch_size*10,numpix_side*2,numpix_side*2,1])
#     NOISE_test = np.zeros(UVGRID_test.shape)
#     read_data_batch(X_test,UVGRID_test,Y_test,NOISE_test,10000,'test')
#
#     y_ = tf.placeholder(dtype=tf.float32,shape=[None,numpix_side**2])
#     y_image = tf.reshape(y_,[-1,numpix_side,numpix_side,1])
#     uvgrid_pl = tf.placeholder(dtype=tf.int32,shape=[None,numpix_side*2,numpix_side*2,1])
#     noise_pl = tf.placeholder(dtype=tf.complex64,shape=[None,numpix_side*2,numpix_side*2,1])
#
#
#     likelihoodobj = ALMA_likelihood2.ALMA_sampler(y_image,uvgrid_pl,noise_pl)
#     x_init = tf.zeros_like(y_image)
#     n_channel = 1
#
#     print tf.shape(x_init)
#     # Needed for Optimization purposes
#     global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
#     is_training = tf.placeholder(tf.bool, [], name='is_training')
#
#     # Number of steps to perform for inference
#     T = tf.constant(FLAGS.t_max, dtype=tf.int32, name='T')
#
#     ## Define some helper functions
#     def param2image(x_param):
#         x_temp = tf.nn.sigmoid(x_param)
#         # x_temp = tf.nn.relu(x_param)
#         return x_temp
#
#     def image2param(x):
#         x_temp = tf.log(x) - tf.log(1 - x)
#         return x_temp
#
#     def param2grad(x_param):
#         x_temp = tf.nn.sigmoid(x_param) * (1. - tf.nn.sigmoid(x_param))
#         return x_temp
#
#     def error_grad(x_test):
#         return tf.map_fn(lambda xin: tf.image.per_image_standardization(xin),tf.gradients(likelihoodobj.loglikelihood(x_test),x_test)[0])
#
#     def lossfun(x_est,expand_dim=False):
#         temp_data = y_image
#         if expand_dim:
#             temp_data = tf.expand_dims(temp_data,0)
#         return tf.reduce_sum(0.5 * tf.square(x_est - temp_data) , [-3,-2,-1] )
#     ## End helper functions
#
#
#     ## Setup RNN
#     if FLAGS.use_rnn:
#         print "Using RNN"
#         cell, output_func = conv_rnn.gru(FLAGS.k_size, [FLAGS.features]*FLAGS.depth, is_training=is_training)
#     else:
#         print "Using Relu network"
#         cell, output_func = conv_rnn.relu(FLAGS.k_size, [FLAGS.features]*FLAGS.depth, is_training=is_training)
#
#
#     ## Defines how the output dimensions are handled
#     output_shape_dict = {'mu':n_channel}
#     if FLAGS.n_pseudo > 0:
#         output_shape_dict.update({'pseudo': n_channel*FLAGS.n_pseudo})
#
#     output_transform_dict = {}
#     if FLAGS.use_prior:
#         output_transform_dict.update({'all':[tf.identity]})
#
#     if FLAGS.use_grad:
#          output_transform_dict.update({'mu': [error_grad]})
#          if FLAGS.n_pseudo > 0:
#             output_transform_dict.update({'pseudo':[loopfun.ApplySplitFunction(error_grad, 4 - 1, FLAGS.n_pseudo)]})
#
#     input_func, output_func, init_func, output_wrapper = decorate_rnn.init(rank=4, output_shape_dict=output_shape_dict,
#                                                                   output_transform_dict=output_transform_dict,
#                                                                   init_name='mu', ofunc = output_func,
#                                                                   accumulate_output=FLAGS.accumulate_output)
#
#     ## This runs the inference
#     x_init_feed = image2param(tf.maximum(tf.minimum(x_init, 1. - 1e-4), 1e-4))
#
#     print x_init_feed , cell , input_func , output_func , init_func , T
#     alltime_output, final_output, final_state, p_t, T_ = \
#         iterative_estimation.function(x_init_feed, cell, input_func, output_func, init_func, T=T,t_max=30)
#
#     final_state = tf.identity(final_state, name='final_state')
#     p_t = tf.identity(p_t, 'p_t')
#     T_ = tf.identity(T_, 'T_')
#
#     alltime_output = param2image(output_wrapper(alltime_output, 'mu', 4))
#     final_output = param2image(output_wrapper(final_output, 'mu'))
#
#     alltime_output = tf.identity(alltime_output,name='alltime_output')
#     final_output = tf.identity(final_output, name='final_output')
#
#     tf.add_to_collection('output', alltime_output)
#     tf.add_to_collection('output', final_output)
#
#     ## Define loss functions
#     loss_full = tf.reduce_sum(tf.reduce_mean(p_t * lossfun(alltime_output, True), reduction_indices=[1]))
#     loss = tf.reduce_mean(lossfun(final_output))
#     tf.add_to_collection('losses', loss_full)
#     tf.add_to_collection('losses', loss)
#
#     psnr = tf.reduce_mean(get_psnr(final_output, y_image))
#     psnr_x_init = tf.reduce_mean(get_psnr(x_init, y_image))
#     tf.add_to_collection('psnr', psnr)
#     tf.add_to_collection('psnr', psnr_x_init)
#
#     ## Minimizer
#     minimize = tf.contrib.layers.optimize_loss(loss_full, global_step, FLAGS.lr, "Adam", clip_gradients=5.0,
#                                                learning_rate_decay_fn=lambda lr,s: tf.train.exponential_decay(lr, s,
#                                                decay_steps=5000, decay_rate=0.96, staircase=True))
#
#     #update_ops = tf.get_collection('batch_norm')
#     #if update_ops:
#     #    updates = tf.tuple(update_ops)
#     #    minimize = control_flow_ops.with_dependencies(updates, minimize)
#
#     # Initializing the variables
#     init_op = tf.global_variables_initializer()
#
#     # Create a summary to monitor cost function
#     loss_summary = tf.summary.scalar("Loss", loss)
#     mse_summary = tf.summary.scalar("PSNR", psnr)
#     # summary_x = tf.image_summary("Corrupted image", x_init, max_images=5)
#     # summary_y = tf.image_summary("Ground truth image", input_data, max_images=5)
#     # summary_pred = tf.image_summary("Reconstructed image", final_output, max_images=5)
#
#     # Merge all summaries to a single operator
#     merged_summary_op = tf.summary.merge_all()
#
#     saver = tf.train.Saver(max_to_keep=None)
#
#     # Launch the graph
#     with tf.Session() as sess:
#
#         sess.run(init_op)
#         # Keep training until reach max iterations
#
#         # Set logs writer into folder /tmp/tensorflow_logs
#
#         saver.restore(sess,os.environ['WORK']+'/trained_weights/CLEAN_RIM/model_new_likelihood2.ckpt')
#
#         for epoch in range(FLAGS.n_epochs):
#             #train_x = X[np.random.permutation(X.shape[0])]
#
#             train_cost = 0.
#             train_psnr = 0.
#
#             print "Sampling data"
#             #train_batches = make_batches(train_x.shape[0], FLAGS.batch_size)
#
#             print "Iterating..."
#             # Loop over all batches
#             for i in range(50000):
#                 tstart = time.time()
#                 #batch_xs = train_x[train_batches[i]:train_batches[i+1]]
#                 read_data_batch(X,UVGRID,Y,NOISE,100000,'train')
#                 # Fit training using batch data
#                 temp_cost, temp_psnr, summary_str,_ = sess.run([loss,psnr,merged_summary_op,minimize],
#                                                                {y_:X,uvgrid_pl:UVGRID,noise_pl:NOISE, is_training:True})
#                 # Compute average loss
#                 train_cost += temp_cost
#                 train_psnr += temp_psnr
#
#                 if i % 100 == 0:
#                     valid_cost = 0.
#                     valid_psnr = 0.
#                     #valid_batches = make_batches(valid_x.shape[0], FLAGS.batch_size)
#                     for j in range(10):
#                         #batch_xs = valid_x[valid_batches[j]:valid_batches[j+1]]
#                         dpm = 2
#                         temp_cost, temp_psnr = sess.run([loss,psnr], {y_:X_test[dpm*j:dpm*(j+1),:],uvgrid_pl:UVGRID_test[dpm*j:dpm*(j+1),:],is_training:False,noise_pl:NOISE_test[dpm*j:dpm*(j+1),:]})
#                         # Compute average loss
#                         valid_cost += temp_cost
#                         valid_psnr += temp_psnr
#
#                     valid_cost /= 10.
#                     valid_psnr /= 10.
#
#                     # Display logs per epoch step
#                     print "Epoch:", '%04d' % (epoch+1), "batch:", '%04d' % (i+1)
#                     print "cost=", "{:.9f}".format(train_cost/(i+1))
#                     print "psnr=", "{:.9f}".format(train_psnr/(i+1))
#                     print "test cost=", "{:.9f}".format(valid_cost)
#                     print "test psnr=", "{:.9f}".format(valid_psnr)
#
#                     # Write predicted and true images
# #                    np.save(os.environ['HOME']+'/RIM/network_outputs/output_images.npy',imgs)
# #                    np.save(os.environ['HOME']+'/RIM/network_outputs/True_images.npy',dataprocessor.Y_test)
#
#                     # Saving Checkpoint
#                     if valid_cost < min_test_cost:
#                         print "Saving Checkpoint"
#                         saver.save(sess,full_path)
#                         min_test_cost = valid_cost
#
#                 tstop = time.time()
#                 if i%10 ==0:
#                     print('Iteration:  %i   took: %.2f seconds.  Cost:  %.5f   pSNR:  %.5f'%(i,tstop-tstart,train_cost/float(i+1),train_psnr/float(i+1)))
# #                    print "cost=", "{:.9f}".format(train_cost/(i+1))
# #                    print "psnr=", "{:.9f}".format(train_psnr/(i+1))
#
#         print "Optimization Finished!"
#
#     sess.close()
#
# def main(argv=None):  # pylint: disable=unused-argument
#   FLAGS.train_dir = path.expanduser(FLAGS.train_dir)
#   tf.gfile.MakeDirs(FLAGS.train_dir)
#   train()
#
#
# if __name__ == '__main__':
#   tf.app.run()
