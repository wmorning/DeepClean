import numpy as np
from PIL import Image
import os,sys
import DeepClean as DC
import struct


# WHY DID WE HAVE TO WRITE IT THIS WAY!!!!!!!!!!!!!!!!
numpix_side = 192
batch_size = 2
pix_res = 0.04
L_side = numpix_side*pix_res
global max_noise_rms, max_psf_rms , max_cr_intensity
max_trainoise_rms = 0.1
max_testnoise_rms = 0.1
max_noise_rms = max_testnoise_rms
cycle_batch_size = 10
num_test_samples = 500
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


xv, yv = np.meshgrid( np.linspace(-L_side/2.0, L_side/2.0, num=numpix_side) ,  np.linspace(-L_side/2.0, L_side/2.0, num=numpix_side))


def read_batch_online( X , Y , max_file_num , train_or_test):
	num_samp = X.shape[0]
	Xmat, Ymat = eng.online_image_generator(num_samp , -1 , numpix_side , os.environ['LOCAL_SCRATCH'] , nargout=2)
	Xmat = np.array(Xmat._data.tolist())
	Xmat = Xmat.reshape((num_samp,numpix_side,numpix_side))
	Xmat = np.transpose(Xmat, axes=(0,2,1)).reshape((num_samp,numpix_side*numpix_side))
	Ymat = np.array(Ymat._data.tolist())
	Ymat = Ymat.reshape((num_out,num_samp)).transpose()
	X[:] = Xmat
	Y[:] = Ymat


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



def gen_masks(nmax,ARCS , apply_prob=0.5):
        mask = 1.0
	if np.min(ARCS)<0.1 and np.max(ARCS)>0.9:
        	if np.random.uniform(low=0, high=1)<=apply_prob:
			while True:
                		mask = np.ones((numpix_side,numpix_side),dtype='float32')
                		num_mask = np.random.randint(1, high = nmax)
                		for j in range(num_mask):
                        		x_mask =  np.random.uniform(low=-L_side/2.0, high=L_side/2.0)
                        		y_mask =  np.random.uniform(low=-L_side/2.0, high=L_side/2.0)
                        		r_mask = np.sqrt( (xv- x_mask  )**2 + (yv- y_mask )**2 )
                        		mask_rad = 0.2
                        		mask = mask * np.float32(r_mask>mask_rad)
				if np.sum(mask*ARCS) >= ( min_unmasked_flux * np.sum(ARCS)):
					break
	return mask


def apply_psf(im , my_max_psf_rms , apply_prob=1.0 ):
        np.random.uniform()
        rand_state = np.random.get_state()
	if np.random.uniform()<= apply_prob:
    		psf_rms = np.random.uniform(low= 0.1 , high=my_max_psf_rms)
    		blurred_im = scipy.ndimage.filters.gaussian_filter( im.reshape(numpix_side,numpix_side) , psf_rms)
		if np.max(blurred_im)!=0:
    			blurred_im = blurred_im / np.max(blurred_im)
    		im[:] = blurred_im.reshape((-1,numpix_side*numpix_side))
	np.random.set_state(rand_state)


def add_poisson_noise(im,apply_prob=1):
	np.random.uniform()
	rand_state = np.random.get_state()
	if np.random.uniform()<= apply_prob:
		intensity_to_photoncounts = np.random.uniform(low=50.0, high=1000.0)
		photon_count_im = np.abs(im * intensity_to_photoncounts)
		poisson_noisy_im = np.random.poisson(lam=(photon_count_im), size=None)
		im_noisy = np.double(poisson_noisy_im)/intensity_to_photoncounts 
		im_noisy = im_noisy/np.max(im_noisy)
		im[:] = im_noisy
	np.random.set_state(rand_state)


def add_cosmic_ray(im,apply_prob=1):
	rand_state = np.random.get_state()
	if np.random.uniform()<= apply_prob:
		inds_cr = np.random.randint(0, high = 400000)
		filename_cr =  CRay_data_path + 'cosmicray_' + "%07d" % (inds_cr+1) + '.png'
		CR_MAP = np.array(Image.open(filename_cr),dtype='float32').reshape(numpix_side*numpix_side,)/255.0
		if np.max(CR_MAP)>0.1 and np.min(CR_MAP)<0.1:
			CR_MAP = CR_MAP/np.max(CR_MAP)
		else:
			CR_MAP = CR_MAP * 0
		CR_SCALE = np.random.uniform(low=0.0, high=max_cr_intensity)
		im[:] = im[:] + (CR_SCALE * CR_MAP)
	np.random.set_state(rand_state)

def pixellation(im_input):
        im = np.max(im_input)
        im =  im.reshape(numpix_side,numpix_side)
        numccdpix = np.random.randint(96, high=192)
        FACTOR = np.float( numccdpix)/192.0
        im_ccd =scipy.ndimage.interpolation.zoom( im , FACTOR )
        im_ccd_max = np.max(im_ccd)
        im_ccd = im_ccd * im_max / im_ccd_max
        add_gaussian_noise(im_ccd)
        im = scipy.ndimage.interpolation.zoom( im_ccd , 1/FACTOR )
        im = im * im_max / np.max(im)
	im_input[:] = im


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

def read_data_batch( X , PSF, Y , noise, max_file_num , train_or_test,returnuv = False,antennaconfig=None):
    batch_size = len(X)
    #mag = np.zeros((batch_size,1))
    if train_or_test=='test':
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
    U = []
    V = []

    for i in range(batch_size):

        #ARCS=1
        #nt = 0

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


				ARCS = np.array(Image.open(arc_filename),dtype='float32').reshape(numpix_side*numpix_side,)/65535.0


		ARCS_SHIFTED, lensXY , m_shift, n_shift = pick_new_lens_center(ARCS,Y[i,:], xy_range = max_xy_range)

		ARCS = np.copy(ARCS_SHIFTED).reshape(numpix_side,numpix_side) 


                if (np.all(np.isnan(ARCS)==False)) and ((np.all(ARCS>=0)) and (np.all(np.isnan(Y[i,3:5])==False))) and ~np.all(ARCS==0):
                        break

        rand_state = np.random.get_state()

	im_telescope = np.copy(ARCS) 
	im_telescope = im_telescope.reshape((numpix_side,numpix_side))


	UVGRID,db,u,v = DC.get_new_UVGRID_and_db(pix_res,numpix_side*2,deposit_order=0,antennaconfig=antennaconfig)

	if np.any(ARCS>0.4):
        	val_to_normalize = np.max(im_telescope[ARCS>0.4])
		if val_to_normalize==0:
			val_to_normalize = 1.0
		int_mult = 1.0-abs(np.random.normal(loc=0.0, scale = 0.01))
        	im_telescope = (im_telescope / val_to_normalize) * int_mult 
	
	noise_scl = np.random.uniform(max_noise_rms/10.,max_noise_rms)
#	noise[i,:] = np.random.normal(0.0,noise_scl*np.max(im_telescope)*np.sqrt(2),UVGRID.shape).reshape(1,2*numpix_side,2*numpix_side,1)
	
	dim_telescope = np.fft.ifft2(np.fft.fft2(np.pad(im_telescope,[[numpix_side/2,numpix_side/2],[numpix_side/2,numpix_side/2]],mode='constant',constant_values=0.))*np.fft.fftshift(UVGRID>0)/192.).real
	noise_realization = np.random.normal(0.0,1.0,UVGRID.shape)
	noise_dty = np.fft.ifft2(np.fft.fft2(noise_realization)/np.sqrt(np.fft.fftshift(UVGRID)+10**-8)*(np.fft.fftshift(UVGRID>0))/384.).real
	noise[i,:] = noise_realization.reshape(1,2*numpix_side,2*numpix_side,1) * np.max(dim_telescope)/np.std(noise_dty)*noise_scl
	
	

        X[i,:] = im_telescope.reshape((1,-1))
	PSF[i,:] = np.fft.fftshift(UVGRID).reshape((1,numpix_side*2,numpix_side*2,1))
       	Y[i,3] = lensXY[0]
       	Y[i,4] = lensXY[1]

	np.random.set_state(rand_state)
	
	if returnuv:
		U.append(u)
		V.append(v)

    if returnuv:
	    return U,V

def read_test_data_batch( X , PSF , Y_test , Y_CLEAN ):
    batch_size = len(X)
    inds = range(batch_size)
    for i in range(batch_size):
        filename = os.environ['SCRATCH'] + "/CASA_IMS/dirty_im_sim_" + "%03d" % (i+1) + '.fits'
        hdulist = fits.open(filename)
        im = hdulist[0].data
	X[i,:] = im.reshape((1,-1))

        filename = os.environ['SCRATCH'] + "/CASA_IMS/psf_sim_" + "%03d" % (i+1) + '.fits'
	hdulist = fits.open(filename)
        im = hdulist[0].data
        PSF[i,:] = im.reshape((1,-1))

        filename = os.environ['SCRATCH'] + "/CASA_IMS/im_" + "%03d" % (i+1) + '.fits'
        hdulist = fits.open(filename)
        im = hdulist[0].data
        Y_test[i,:] = im.reshape((1,-1))

        filename = os.environ['SCRATCH'] + "/CASA_IMS/clean_im_sim_" + "%03d" % (i+1) + '.fits'
        hdulist = fits.open(filename)
        im = hdulist[0].data
        Y_CLEAN[i,:] = im.reshape((1,-1))



#file_list = []
#n = 0
#for file in os.listdir( os.environ['SCRATCH'] + "/CLEAN_DATA/fits_DIM" ):
#    if file.startswith("dirty_im"):
#        #print(file)
#	file_list.insert( 0 , file)
#	n = n + 1

#def read_train_data_batch( X , PSF , Y_test  ):
#    batch_size = len(X)
#    max_file_num = len(file_list)
#    inds = range(batch_size)
#    for i in range(batch_size):
#	pick_file = np.random.randint(0, high = max_file_num)
#	file_num = file_list[pick_file][13:19]

#        filename = os.environ['SCRATCH'] + '/CLEAN_DATA/fits_DIM/dirty_im_sim_' + file_num + '.fits'
#        hdulist = fits.open(filename)
#        im = hdulist[0].data

#        filename = os.environ['SCRATCH'] + '/CLEAN_DATA/fits_DIM/dirty_noise_sim_' + file_num + '.fits'
#        hdulist = fits.open(filename)
#        noise = hdulist[0].data

#	im = im / np.max(im)
#	im = im + np.random.uniform(low=max_noise_rms/100, high=max_noise_rms) * noise
#        X[i,:] = im.reshape((1,-1))

#        filename = os.environ['SCRATCH'] + '/CLEAN_DATA/fits_PSF/psf_sim_' + file_num + '.fits'
#        hdulist = fits.open(filename)
#        im = hdulist[0].data
#        PSF[i,:] = im.reshape((1,-1))

#        filename = os.environ['SCRATCH'] + '/CLEAN_DATA/sky_images/im_' + file_num + '.fits'
#        hdulist = fits.open(filename)
#        im = hdulist[0].data
#        Y_test[i,:] = im.reshape((1,-1))



#def read_CASA_test_data_batch( X , PSF , Y_test , Y_CLEAN ):
#    batch_size = len(X)
#    inds = range(batch_size)
#    for i in range(batch_size):
#        filename = os.environ['SCRATCH'] + "/CASA_IMS/im_" + "%03d" % (i+1) + '.fits'
#        hdulist = fits.open(filename)
#        im = hdulist[0].data

#	im = im.reshape((-1,1))
#        im_telescope = np.copy(im)
#        im_telescope = im_telescope.reshape((numpix_side,numpix_side))

#        FT_im =  np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im_telescope)) )

#        UV_im =  Nature_W * FT_im
#        DT_im =  np.real( np.fft.ifftshift( np.fft.ifft2( np.fft.ifftshift(UV_im) ) ) )
#        DT_im = DT_im / np.max(DT_im)

#        DT_BEAM =  np.real( np.fft.ifftshift( np.fft.ifft2( np.fft.ifftshift(UVGRID) ) ) )

#        FFT_NOISE = (np.random.normal(loc=0.0, scale = np.sqrt(Nvar) )  + np.random.normal(loc=0.0, scale = np.sqrt(Nvar)  ) *1j)

#        noise_map = make_real_noise(FFT_NOISE)

#        rnd_noise_rms = np.random.uniform(low=max_noise_rms/100, high=max_noise_rms)
#        dirty_im_noisy = DT_im + (rnd_noise_rms * noise_map)
#        im_telescope = dirty_im_noisy
#        im_telescope = im_telescope.reshape((-1,1))

#        if np.any(im>0.4):
#                val_to_normalize = np.max(im_telescope[im>0.4])
#                if val_to_normalize==0:
#                        val_to_normalize = 1.0
#                int_mult = np.random.normal(loc=1.0, scale = 0.01)
#                im_telescope = (im_telescope / val_to_normalize) * int_mult


#        X[i,:] = im_telescope.reshape((1,-1))
#        PSF[i,:] = DT_BEAM.reshape((1,-1))
#        Y_test[i,:] = im.reshape((1,-1))


#        filename = os.environ['SCRATCH'] + "/CASA_IMS/clean_im_sim_" + "%03d" % (i+1) + '.fits'
#        hdulist = fits.open(filename)
#        im = hdulist[0].data
#        Y_CLEAN[i,:] = im.reshape((1,-1))



#def read_SLACS_data_batch( X , Y ):
#    batch_size = len(X)
#    for i in range(12):
#        filename1 = os.environ['SCRATCH'] + "/SLACS_txt/test_0000001_1.txt"
#        im =  np.loadtxt(filename1)
#        X[i,:] = im.reshape((1,-1))
#    Y_all = np.loadtxt(os.environ['SCRATCH'] + "/SLACS_txt/parameters_test.txt").reshape((1,-1))
#    Y[:,:] = np.matlib.repmat(Y_all[:,0:5], 12 , 1)










def gen_circle( X , Y ):
	coord_rot_angle = -45 * np.pi/180
	R = np.array([ [cos(coord_rot_angle) , sin(coord_rot_angle)] , [-sin(coord_rot_angle) , cos(coord_rot_angle)] ] )
	R = 1.0
	ex = np.array([1,0]).reshape([2,1])
	ey = np.array([0,1]).reshape([2,1])
	Tey = np.matmul(R, ey)
	T = np.concatinate( (ex,ey) , axis=1 )
	tilted_coords = np.concatinate( ( X.reshape((1,-1)) , Y.reshape((1,-1)) ) , axis=0 )

	orthog_coords = np.matmul( T , tilted_coords );
	x_orthog = orthog_coords[0,:] 
	y_orthog = orthog_coords[1,:]

	batch_size = len(x_new)
	circle = 1.0
	for i in range(batch_size):
                r_c = np.sqrt( (xv- x_orthog[i]  )**2 + (yv- y_orthog[i] )**2 )
                circle_rad = 0.2
                circle = circle * np.float32(r_c<=circle_rad)

		X[i,:] = circle.reshape((1,-1))
		Y[i,:] = np.array([x_1 , x_2 , y_1 , y_2])
       






#def write_im_fits( N ):
#    batch_size = N
#    inds = range(batch_size)
#    np.random.seed(seed=2)
#    d_path = [[],[]]
#    d_path[0] = test_data_path_1
#    d_path[1] = test_data_path_2

#    for i in range(batch_size):

#	print i
#        while True:
#                ARCS=1
#                nt = 0
#                while np.min(ARCS)==1 or np.max(ARCS)<0.4:
#                        nt = nt + 1
#                        if nt>1:
#                                inds[i] = np.random.randint(0, high = max_file_num)



#                        pick_folder = np.random.randint(0, high = num_data_dirs)
#                        arc_filename = d_path[pick_folder] +  'train_' + "%07d" % (inds[i]+1) + '.png'



#                        ARCS = np.array(Image.open(arc_filename),dtype='float32').reshape(numpix_side*numpix_side,)/65535.0

                #ARCS_SHIFTED, lensXY , m_shift, n_shift = pick_new_lens_center(ARCS,Y[i,:], xy_range = max_xy_range)
                #ARCS = np.copy(ARCS_SHIFTED)


#                if (np.all(np.isnan(ARCS)==False)) and (np.all(ARCS>=0)) and ~np.all(ARCS==0):
#                        break


#        im_telescope = np.copy(ARCS)
#        im_telescope = im_telescope.reshape((numpix_side,numpix_side))



#        val_to_normalize = np.max(im_telescope)
#        if val_to_normalize==0:
#            val_to_normalize = 1.0
#        im_telescope = (im_telescope / val_to_normalize) 

#	hdu = fits.PrimaryHDU(im_telescope)
#	hdulist = fits.HDUList([hdu])
#	hdulist.writeto( os.environ['LOCAL_SCRATCH'] +  '/sky_images/im_' +  "%0.6d"%i  +'.fits')

def load_binary(binaryfile):
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
    
    kvec = np.fft.fftshift(np.fft.fftfreq(num_pixels,pix_res/3600./180.*np.pi))
    kvec -= (kvec[1]-kvec[0])/2.
    kvec = np.append(kvec,kvec[-1]+(kvec[1]-kvec[0]))
    
    print np.sum(np.isclose((kvec[1:]+kvec[:-1])/2.,0))
    
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
    
    
    # loop over bins
    for i in range(len(row)):
        
        # indices of visibilities in the bin
        inds = np.where((v>=kvec[col[i]]) & (v<kvec[col[i]+1]) & \
                        (u>=kvec[row[i]]) & (u<kvec[row[i]+1]))[0]
        

        
        vis_gridded[col[i],row[i]] +=np.sum(vis[inds])
    
    for i in range(len(row2)):
        
        # indices of visibilities in the bin
        inds = np.where((-v>=kvec[col2[i]]) & (-v<kvec[col2[i]+1]) & \
                        (-u>=kvec[row2[i]]) & (-u<kvec[row2[i]+1]))[0]
        
    
        
        vis_gridded[col2[i],row2[i]] +=np.sum(np.conj(vis[inds]))
    
    # get average by division
    #vis_gridded[np.where(P+P2 !=0)] /= (P+P2)[np.where(P+P2 !=0)].astype('float')
    
    vis_gridded /= (P.T+P2.T+1e-8)
    vis_gridded[np.abs(vis_gridded)<1e-6] *=0

    return (P+P2).T , vis_gridded

def get_gridded_visibilities(directory_name,pix_res,num_pixels,phasecenter=[0.,0.],newRipples=False):
	'''
	Load visibilities from a file, and then produce the gridded (averaged in grid cells) 
	visibilities and uv mask that can be fed to the likelihood object)
	'''
	# first lets load the data
	u = load_binary(directory_name+'u.bin')
	v = load_binary(directory_name+'v.bin')
	vis = load_binary(directory_name+'vis_chan_0.bin')
	vis = vis[::2]+1j*vis[1::2]

	if newRipples is True:
		freq = load_binary(directory_name+'frequencies.bin')
		wav =  (3.*10**8) / freq
		u /= wav
		v /= wav
	print(np.max(np.sqrt(u**2+v**2)))
	vis = shift_phase_center(u,v,vis,phasecenter)
	print(np.max(np.abs(vis)))
	UVGRID , vis_gridded = get_binned_visibilities(u,v,vis,pix_res,num_pixels)

	return np.fft.fftshift(UVGRID).reshape([1,num_pixels,num_pixels,1]) , np.fft.fftshift(vis_gridded).reshape([1,num_pixels,num_pixels,1])


def shift_phase_center(u,v,vis,phase_center):
	'''
	Shift the center of the ALMA pointing to a new phase center.  Phase center shift 
	is defined in arcseconds.
	'''
	ushift = phase_center[0]*u / 3600. / 180. * np.pi
	vshift = phase_center[1]*v / 3600. / 180. * np.pi
	phaseshift = ushift + vshift

	vis_shifted = vis * np.exp(2j*np.pi*phaseshift)
	return vis_shifted
