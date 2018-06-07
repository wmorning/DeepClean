import numpy as np
from PIL import Image
import os,sys
from sim_uv_cov import get_new_UVGRID_and_db
import struct
import glob


# If we're importing functions from here, then we need all of this defined locally...
numpix_side = 192
batch_size = 2
pix_res = 0.04
L_side = numpix_side*pix_res
global max_noise_rms, max_psf_rms , max_cr_intensity
max_trainoise_rms = 0.1
max_testnoise_rms = 0.1
max_noise_rms = max_testnoise_rms
cycle_batch_size = 10
num_test_samples = 1000
global arcs_data_path_1, arcs_data_path_2 , test_data_path_1 , test_data_path_2 , CRay_data_path
global lens_data_path_1, lens_data_path_2, testlens_data_path_1, testlens_data_path_2
global min_unmasked_flux
min_unmasked_flux = 0.75
global num_data_dirs
num_data_dirs = 2
num_training_samples = 100000
max_num_test_samples = 1000
arcs_data_path_1 = os.environ['WORK'] + '/NAZGUL/ARCS_1/'
arcs_data_path_2 = os.environ['WORK'] + '/NAZGUL/ARCS_2/'
test_data_path_1 = os.environ['WORK'] + '/NAZGUL/ARCS_1/'
test_data_path_2 = os.environ['WORK'] + '/NAZGUL/ARCS_2/'
max_xy_range=2.0


Y_all_train=[[],[]]
Y_all_test =[[],[]]

Y_all_train[0] = np.loadtxt(arcs_data_path_1 + '/parameters_train.txt')
Y_all_test[0] = np.loadtxt(test_data_path_1 + '/parameters_test.txt')
Y_all_train[1] = np.loadtxt(arcs_data_path_2 + '/parameters_train.txt')
Y_all_test[1] = np.loadtxt(test_data_path_2 + '/parameters_test.txt')

R_n = np.loadtxt( os.environ['WORK'] + '/DATA/PS_4_real.txt')
I_n = np.loadtxt( os.environ['WORK'] + '/DATA/PS_4_imag.txt')

CLEAN_data_paths = [os.environ['WORK']+'/DATA/RIM/',\
		    os.environ['WORK']+'/DATA/RIM2/',\
		    os.environ['WORK']+'/DATA/RIM3/',\
		    os.environ['WORK']+'/DATA/RIM4/',\
		    os.environ['WORK']+'/DATA/RIM5/',\
		    os.environ['WORK']+'/DATA/RIM6/',\
		    os.environ['WORK']+'/DATA/RIM7/',\
		    os.environ['WORK']+'/DATA/RIM8/',\
		    os.environ['WORK']+'/DATA/RIM9/']


Y_CLEAN_train = [[],[],[],[],[],[],[],[],[],[]]
for i in range(9):
	Y_CLEAN_train[i]=np.load(CLEAN_data_paths[i]+'parameters_train.npy')




xv, yv = np.meshgrid( np.linspace(-L_side/2.0, L_side/2.0, num=numpix_side) ,  np.linspace(-L_side/2.0, L_side/2.0, num=numpix_side))



def make_real_noise(Fmap):
    Npix = Fmap.shape[0];
    Npix_2 = Npix/2;
    Npix_2p1 = Npix/2 + 1;
    Npix_2p2 = Npix/2 + 2;
    Npix_2m1 = Npix/2 - 1;

    A = np.concatenate( (Fmap[0:Npix_2,Npix_2p2-1:] , np.conj(np.fliplr(np.flipud(Fmap[Npix_2p1-1,1:Npix_2].reshape((1,-1))))) ) , axis = 0)
    B = np.concatenate( (Fmap[0:Npix_2p1,0:Npix_2p1], A) , axis = 1)
    C = np.concatenate( (np.zeros((Npix_2m1,1)) , np.conj(np.fliplr(np.flipud(Fmap[1:Npix_2,Npix_2p2-1:]))), np.conj(np.fliplr(np.flipud(Fmap[1:Npix_2,1:Npix_2p1]))) ) , axis = 1)
    sym_fft = np.concatenate( (B ,C ) , axis = 0)
    noise_map = np.real( np.fft.ifft2(np.fft.ifftshift(sym_fft)) )
    noise_map = noise_map/np.std(noise_map)
    return noise_map


def add_gaussian_noise(im):
    if variable_noise_rms == False:
    	rnd_noise_rms=max_noise_rms
    else:
	rnd_noise_rms = np.random.uniform(low=max_noise_rms/10, high=max_noise_rms)

    if np.random.uniform(low=0, high=1)<=1.0:
    	noise_map = np.random.normal(loc=0.0, scale = rnd_noise_rms,size=im.shape)
    else:
	FFT_NOISE = np.random.normal(loc=0.0, scale = np.abs(R_n))  + np.random.normal(loc=0.0, scale = np.abs(I_n) ) *1j
	noise_map = make_real_noise(FFT_NOISE)
    	noise_map = rnd_noise_rms * noise_map
	noise_map = noise_map.reshape((1,-1))
    im[:] = im[:] + noise_map




def im_shift(im, m , n):
    shifted_im1 = np.zeros(im.shape)
    if n > 0:
        shifted_im1[n:,:] = im[:-n,:]
    elif n < 0:
        shifted_im1[:n,:] = im[-n:,:]
    elif n ==0:
        shifted_im1[:,:] = im[:,:]
    shifted_im2 = np.zeros(im.shape)
    if m > 0:
        shifted_im2[:,m:] = shifted_im1[:,:-m]
    elif m < 0:
        shifted_im2[:,:m] = shifted_im1[:,-m:]
    shifted_im2[np.isnan(shifted_im2)] = 0
    return shifted_im2

def pick_new_lens_center(ARCS,Y, xy_range = 0.5):
	rand_state = np.random.get_state()
	while True:
        	x_new = np.random.randint( -1 * np.ceil(xy_range/2/pix_res) , high = np.ceil(xy_range/2/pix_res) )
        	y_new = np.random.randint( -1 * np.ceil(xy_range/2/pix_res) , high = np.ceil(xy_range/2/pix_res) )
        	m_shift = - int(np.floor(Y[3]/pix_res) - x_new)
        	n_shift = - int(np.floor(Y[4]/pix_res) - y_new)
        	shifted_ARCS = im_shift(ARCS.reshape((numpix_side,numpix_side)), m_shift , n_shift ).reshape((numpix_side*numpix_side,))
		if np.sum(shifted_ARCS) >= ( 0.98 * np.sum(ARCS) ):
			break
        #lensXY = np.array( [ np.double(x_new) * pix_res+ (Y[3]%pix_res) , np.double(y_new) * pix_res + (Y[4]%pix_res) ])
	lensXY = np.array( [ np.double(m_shift) * pix_res+ Y[3] , np.double(n_shift) * pix_res + Y[4] ])
	np.random.set_state(rand_state)
	return shifted_ARCS , lensXY , m_shift, n_shift


def read_data_batch( X , PSF, Y , noise, max_file_num , train_or_test,uvconfig=None):
    '''
    Read a batch of data.  Can be either train or test data.
    '''
    batch_size = len(X)
    if train_or_test=='test': # if test data, it should be deterministic.  So lets use a random seed.
        inds = range(batch_size)
        np.random.seed(seed=2)
	d_path = [[],[]]
	d_path[0] = test_data_path_1
	d_path[1] = test_data_path_2
	

    else:
        np.random.seed(seed=None)
        inds = np.random.randint(0, high = max_file_num , size= batch_size)
	d_path = [[],[]]
        d_path[0] = arcs_data_path_1
        d_path[1] = arcs_data_path_2

    image_container = np.zeros([numpix_side,numpix_side])

    for i in range(batch_size):

	while True:
        	ARCS=1
        	nt = 0
        	while np.min(ARCS)==1 or np.max(ARCS)<0.4:
                	nt = nt + 1
			if nt>1:
				inds[i] = np.random.randint(0, high = max_file_num)



			pick_folder = np.random.randint(0, high = num_data_dirs)
			arc_filename = d_path[pick_folder] +  train_or_test + '_' + "%07d" % (inds[i]+1) + '.png'
			if os.path.isfile(arc_filename):
				if train_or_test=='test':
					Y[i,0:8] = Y_all_test[pick_folder][inds[i],0:8]
					Y[i,7] = Y[i,7]/16.

				else:
					Y[i,0:8] = Y_all_train[pick_folder][inds[i],0:8]
					Y[i,7] = Y[i,7]/16.

				# get the image from the disk
				ARCS = np.array(Image.open(arc_filename),dtype='float32').reshape(numpix_side*numpix_side,)/65535.0

		# Randomly shift the position of the lens to boost the training volume.
		ARCS_SHIFTED, lensXY , m_shift, n_shift = pick_new_lens_center(ARCS,Y[i,:], xy_range = max_xy_range)

		ARCS = np.copy(ARCS_SHIFTED).reshape(numpix_side,numpix_side) 


                if (np.all(np.isnan(ARCS)==False)) and ((np.all(ARCS>=0)) and (np.all(np.isnan(Y[i,3:5])==False))) and ~np.all(ARCS==0):
                        break

	rand_state = np.random.get_state()

	im_telescope = np.copy(ARCS) 
	im_telescope = im_telescope.reshape((numpix_side,numpix_side))
	
	# Generate an ALMA configuration
	UVGRID,db = get_new_UVGRID_and_db(pix_res,numpix_side*2,deposit_order=0,antennaconfig=uvconfig)

	if np.any(ARCS>0.4):
        	val_to_normalize = np.max(im_telescope[ARCS>0.4])
		if val_to_normalize==0:
			val_to_normalize = 1.0
		int_mult = np.random.normal(loc=1.0, scale = 0.01)
        	im_telescope = (im_telescope / val_to_normalize) * int_mult 
	
	# Draw a scaling for the noise.
	noise_scl = np.random.uniform(max_noise_rms/100.,max_noise_rms)
	
	# To actually scale the noise, we need to create the image, and then get the other components of the scaling.
	dim_telescope = np.fft.ifft2(np.fft.fft2(np.pad(im_telescope,[[numpix_side/2,numpix_side/2],[numpix_side/2,numpix_side/2]],mode='constant',constant_values=0.))*np.fft.fftshift(UVGRID>0)).real
	noise_realization = np.random.normal(0.0,1.0,UVGRID.shape)

	# This is now the correlated noise that would be present in the ALMA observation.
	noise_dty = np.fft.ifft2(np.fft.fft2(noise_realization)/np.sqrt(np.fft.fftshift(UVGRID)+10**-8)*(np.fft.fftshift(UVGRID>0))).real
	# scale the noise and add it to the outputs
	noise[i,:] = noise_realization.reshape(1,2*numpix_side,2*numpix_side,1) * np.max(dim_telescope)/np.std(noise_dty) / np.sqrt(2) * noise_scl
	
	# add the image to the outputs
        X[i,:] = im_telescope.reshape((1,-1))

	# add the uv sampling to the outputs
	PSF[i,:] = np.fft.fftshift(UVGRID).reshape((1,numpix_side*2,numpix_side*2,1))
       	
	# update the shifted position of the lens.
	Y[i,3] = lensXY[0]
       	Y[i,4] = lensXY[1]

	np.random.set_state(rand_state)



def read_test_data(uvconfig=None):

    #mag = np.zeros((batch_size,1))                                                                                                    
    train_or_test = 'test'

    np.random.seed(seed=2)
    d_path = [[],[]]
    d_path[0] = test_data_path_1
    d_path[1] = test_data_path_2
    bad_data1 = np.loadtxt(os.environ['WORK']+'/NAZGUL/ARCS_1/arcs1_ignore.txt')
    bad_data2 = np.loadtxt(os.environ['WORK']+'/NAZGUL/ARCS_2/arcs2_ignore.txt')

    
    image_container = np.zeros([numpix_side,numpix_side])
    
    fileslist = []
    for i in range(1000):
	    # go through all files and select the ones that are acceptable
	    if not i+1 in bad_data1:
		    fileslist.append([0,i+1])
	    else:
		    pass
	    if not i+1 in bad_data2:
		    fileslist.append([1,i+1])
	    else:
		    pass

    batch_size = len(fileslist) 
    X = np.zeros([batch_size,numpix_side**2])
    Y = np.zeros([batch_size,num_out])
    PSF = np.zeros([batch_size,2*numpix_side,2*numpix_side,1])
    noise = np.zeros([batch_size,2*numpix_side,2*numpix_side,1])
    inds = range(batch_size)
    
    print "batch size:  " , batch_size

    for i in range(batch_size):
	    print("image {0} out of {1}".format(i,batch_size))
	    while True:
                ARCS=1
                nt = 0
                while np.min(ARCS)==1 or np.max(ARCS)<0.4:
                        nt = nt + 1
                        if nt>1:
                                #inds[i] = np.random.randint(0, high = max_file_num)
				
				print nt
				
                        #pick_folder = np.random.randint(0, high = num_data_dirs)
                        arc_filename = d_path[fileslist[i][0]] +  train_or_test + '_' + "%07d" % (fileslist[i][1]) + '.png'
			print arc_filename
                        if os.path.isfile(arc_filename):
                                if train_or_test=='test':
                                        Y[i,0:8] = Y_all_test[fileslist[i][0]][fileslist[i][1]-1,0:8]
                                        Y[i,7] = Y[i,7]/16.

#                                else:
#                                        Y[i,0:8] = Y_all_train[pick_folder][inds[i],0:8]
#                                        Y[i,7] = Y[i,7]/16.


                                ARCS = np.array(Image.open(arc_filename),dtype='float32').reshape(numpix_side*numpix_side,)/65535.0


                ARCS_SHIFTED, lensXY , m_shift, n_shift = pick_new_lens_center(ARCS,Y[i,:], xy_range = max_xy_range)

                ARCS = np.copy(ARCS_SHIFTED).reshape(numpix_side,numpix_side)


                if (np.all(np.isnan(ARCS)==False)) and ((np.all(ARCS>=0)) and (np.all(np.isnan(Y[i,3:5])==False))) and ~np.all(ARCS==0):
                        break

	    #rand_state = np.random.get_state()

	    im_telescope = np.copy(ARCS)
	    im_telescope = im_telescope.reshape((numpix_side,numpix_side))

	    UVGRID,db = get_new_UVGRID_and_db(pix_res,numpix_side*2,deposit_order=0,antennaconfig=uvconfig)

	    if np.any(ARCS>0.4):
		    val_to_normalize = np.max(im_telescope[ARCS>0.4])
		    if val_to_normalize==0:
			    val_to_normalize = 1.0
		    int_mult = np.random.normal(loc=1.0, scale = 0.01)
		    im_telescope = (im_telescope / val_to_normalize) * int_mult

	    noise_scl = np.random.uniform(max_noise_rms/100.,max_noise_rms)
#           noise[i,:] = np.random.normal(0.0,noise_scl*np.max(im_telescope)*np.sqrt(2),UVGRID.shape).reshape(1,2*numpix_side,2*numpix_side\,1)                                                                                                                                     

	    dim_telescope = np.fft.ifft2(np.fft.fft2(np.pad(im_telescope,[[numpix_side/2,numpix_side/2],[numpix_side/2,numpix_side/2]],mode='constant',constant_values=0.))*np.fft.fftshift(UVGRID>0)).real
	    noise_realization = np.random.normal(0.0,1.0,UVGRID.shape)
	    noise_dty = np.fft.ifft2(np.fft.fft2(noise_realization)/np.sqrt(np.fft.fftshift(UVGRID)+10**-8)*(np.fft.fftshift(UVGRID>0))).real
	    noise[i,:] = noise_realization.reshape(1,2*numpix_side,2*numpix_side,1) * np.max(dim_telescope)/np.std(noise_dty) / np.sqrt(2) * noise_scl

	    

	    X[i,:] = im_telescope.reshape((1,-1))
	    PSF[i,:] = np.fft.fftshift(UVGRID).reshape((1,numpix_side*2,numpix_side*2,1))
	    Y[i,3] = lensXY[0]
	    Y[i,4] = lensXY[1]

	    #np.random.set_state(rand_state)
	    
	    print np.max(X[i,:]),PSF[i,:].shape

    return X,Y,PSF,noise

def load_binary(binaryfile):
	'''
	We find it convenient to store our ALMA datasets as binary files.
	For this reason, it is nice to have a function to read them quickly.
	'''
	with open(binaryfile,'rb') as file:
		filecontent = file.read()
		data = np.array(struct.unpack("d"*(len(filecontent)//8),filecontent))
	file.close()
	return data

def get_binned_visibilities(u,v,vis,pix_res,num_pixels):
    '''
    convert from 1d vector of u, v, vis to a 2d histogram of uv, vis 
    
    Takes:
    
    u:     The u coordinates of the data (in meters)
    
    v:     The v coordinates of the data (in meters)
    
    vis:   The visibility data (in Jy), complex format
    
    Returns:
    
    A:     The noise scaling.
    
    '''
    # these are the coordinates of the edges of grid cells in Fourier space.
    kvec = np.fft.fftshift(np.fft.fftfreq(num_pixels,pix_res/3600./180.*np.pi))
    kvec -= (kvec[1]-kvec[0])/2.
    kvec = np.append(kvec,kvec[-1]+(kvec[1]-kvec[0]))
    
    # Count number of visibilities in each bin
    P,reject1,reject2 = np.histogram2d(u,v,bins=kvec)
    P2,reject1,reject2 = np.histogram2d(-u,-v,bins=kvec)
    vis_gridded = np.zeros(P.shape,dtype=complex)
    
    # Keep only bins that contain visibilities
    [row,col] = np.where(P!=0)
    [row2,col2] = np.where(P2!=0)
    
    # Keep track of stats (just in case something weird is happening)
    NumSkippedBins = 0
    TotalUsed      = 0
    
    # Array for the indices of the visibilities that are subtracted
    indI = np.zeros(u.shape,int)
    
    
    # loop over bins... yes I know its slow.
    for i in range(len(row)):
        
        # indices of visibilities in the bin
        inds = np.where((v>=kvec[col[i]]) & (v<kvec[col[i]+1]) & \
                        (u>=kvec[row[i]]) & (u<kvec[row[i]+1]))[0]
        
        vis_gridded[col[i],row[i]] +=np.sum(vis[inds])
    
    for i in range(len(row2)): # we secretly also sample visibilities at (-u , -v) 
        
        # indices of visibilities in the bin
        inds = np.where((-v>=kvec[col2[i]]) & (-v<kvec[col2[i]+1]) & \
                        (-u>=kvec[row2[i]]) & (-u<kvec[row2[i]+1]))[0]
        
        vis_gridded[col2[i],row2[i]] +=np.sum(np.conj(vis[inds]))
    
    # get average by division.  Avoid NaNs.    
    vis_gridded /= (P.T+P2.T+1e-8)
    vis_gridded[np.abs(vis_gridded)<1e-6] *=0

    return (P+P2).T , vis_gridded

def get_gridded_visibilities(directory_name,pix_res,num_pixels):
	'''
	Load visibilities from a file, and then produce the gridded (averaged in grid cells) 
	visibilities and uv mask that can be fed to the likelihood object)
	'''
	# first lets load the data
	u = load_binary(directory_name+'u.bin')
	v = load_binary(directory_name+'v.bin')
	vis = load_binary(directory_name+'vis_chan_0.bin')
	vis = vis[::2]+1j*vis[1::2]

	# grid the visibilities
	UVGRID , vis_gridded = get_binned_visibilities(u,v,vis,pix_res,num_pixels)
	
	return np.fft.fftshift(UVGRID).reshape([1,num_pixels,num_pixels,1]) , np.fft.fftshift(vis_gridded).reshape([1,num_pixels,num_pixels,1])



def read_CLEAN_data_batch( X , Y , max_file_num ):
	for i in range(X.shape[0]):
		dir_num = np.random.randint(0,9)
		file_num = np.random.randint(0,max_file_num)

		X[i,:] = np.load(CLEAN_data_paths[dir_num]+'train_%07d.npy'%(file_num)).reshape(numpix_side*numpix_side)
		Y[i,:8] = Y_CLEAN_train[dir_num][file_num,:]
