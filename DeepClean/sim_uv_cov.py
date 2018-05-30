'''
Author:  Warren R. Morningstar

A simple-ish script that can do a few things.

1)  For the sake of early Ensai ALMA work, can process a uv configuration
into the proper format.  This circumvents an annoyance from early Ensai models
that required a matlab code to be called to get the uv points processed.

2)  Can simulate a uv coverage either given a configuration file (casa format) 
or by generating its own ALMA configuration.  Right now the distribution of 
baselines used in generating configurations is hard coded (that is to say
we used explicitly specified distributions so as to get a substantial range
of properties in our training sets.  Future iterations will allow this to 
be specified.

3)  Given a list of uv points, create the necessary uv grid that is fed 
to the networks.  For historical reasons it also creates a convolution
kernel for the dirty beam.  The newest codes we have don't use this anymore.
'''
import numpy as np
from itertools import combinations


# some variables that will not change can be defined here
c = 299792458.
ALMA_latitude = -23.02 * np.pi/180.
ALMA_latitude2 =  np.pi / 2. - ALMA_latitude
Rotation_matrix = np.array([[1,0,0],\
                            [0,np.cos(ALMA_latitude2),-np.sin(ALMA_latitude2)],\
                            [0,np.sin(ALMA_latitude2),np.cos(ALMA_latitude2)]])


def cart2pol(x,y):
    '''
    convert from x,y to r,theta
    '''
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(x,y)
    return theta,r

def pol2cart(r,theta):
    '''
    convert from r,theta to x,y
    '''
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x,y

def sim_uv_cov(uvfiles,uvscl=1.0,pixel_size=0.05,Num_pixels=192,randomuv = False,dirtybeam=None):
    num_files = len(uvfiles)
    UVGRIDS = np.zeros([num_files,Num_pixels,Num_pixels])
    dbs = np.zeros([num_files,2*Num_pixels,2*Num_pixels],complex)
    for file_num in range(num_files):
        try:
            uv = np.loadtxt(uvfiles[file_num])
        except:
            uv = np.loadtxt('/scratch/users/hezaveh/CASA_IMS/UV_sim_001.txt')
        #k_max = 0.5/(pixel_size*np.pi/3600./180.)
        #kvec = np.linspace(-k_max,k_max,Num_pixels+1)
        kvec = np.fft.fftshift(np.fft.fftfreq(Num_pixels,pixel_size/3600./180.*np.pi))
        kvec -= (kvec[1]-kvec[0])/2.
        kvec = np.append(kvec,kvec[-1]+(kvec[1]-kvec[0]))
    
        u = uv[:,0]  * uvscl
        v = uv[:,1]  * uvscl
    
        if randomuv is True:
            TH,RHO = cart2pol(u,v)
            theta = (np.random.random()-0.5)*2*np.pi
            u,v = pol2cart(RHO,TH+theta)

        # histogram2d takes y,x to get things right....
        UVGRID,edges,centers = np.histogram2d(v,u,bins=kvec)

        visibility_noise_sigma = 1
        noise_variance = 1. / (np.sqrt(UVGRID) *visibility_noise_sigma)**2
        noise_variance[np.logical_not(np.isfinite(noise_variance))] = 0.0
        Nature_W = 1. / noise_variance
        Nature_W[np.logical_not(np.isfinite(Nature_W))] = 0.0

        # Also, just in case, lets just make an image of the dirty beam (for convolutions)
        if dirtybeam is not None:
            db = np.fft.fft2(np.fft.fftshift(np.load(dirtybeam[file_num]),axes=(0,1)))
        else:
            print "Building the dirty beam:  "
            db = np.zeros([Num_pixels*2,Num_pixels*2])
            xcoords ,ycoords = np.meshgrid(np.arange(-Num_pixels*pixel_size,Num_pixels*pixel_size,pixel_size),\
                                   np.arange(-Num_pixels*pixel_size,Num_pixels*pixel_size,pixel_size))
            xcoords *= np.pi/3600./180.
            ycoords *= np.pi/3600./180.
            for i in range(db.shape[0]):
                print "{0}..".format(i)
                for j in range(db.shape[1]):
                    db[i,j] = np.sum(np.exp(2j*np.pi*(u*xcoords[i,j]+v*ycoords[i,j]))).real

            # To make convolutions less computations, lets take fft and fftshift of the dirty beam
            db = np.fft.fft2(np.fft.fftshift(db,axes=(0,1)))

        dbs[file_num,:,:] = db
        UVGRIDS[file_num,:,:]=UVGRID
    
    return UVGRIDS , noise_variance , Nature_W , u , v , dbs


def grid_deposit(u,v,pixscale,num_pixels,order=1):
    '''                                                                                                                      
    Deposit u and v into an array with num_pixels, corresponding                                                             
    to an angular size given by pixscale.                                                                                    
    '''

    # first, lets get our vector of uv pixels                                                                                
    kvec = np.fft.fftshift(np.fft.fftfreq(num_pixels,pixscale/3600./180.*np.pi))
    dk = kvec[1]-kvec[0]
    
    if order ==1:
        # Now lets get the floating index of each point   
        ui = (u[np.logical_and.reduce((u>np.min(kvec),u<np.max(kvec),v>np.min(kvec),v<np.max(kvec)))] - np.min(kvec))/dk
        vi = (v[np.logical_and.reduce((u>np.min(kvec),u<np.max(kvec),v>np.min(kvec),v<np.max(kvec)))] - np.min(kvec))/dk
    
        # Now, lets setup the corresponding UVGRID (starting in 1-dim)                                                           
        UVGRID = np.zeros(num_pixels**2)

        # Cloud in cell deposit (better than 2d hist)                                                                            
        np.add.at(UVGRID,num_pixels*np.floor(vi).astype(int)+np.floor(ui).astype(int),(1-vi%1)*(1-ui%1))
        np.add.at(UVGRID,num_pixels*np.floor(vi).astype(int)+np.ceil(ui).astype(int),(1-vi%1)*(ui%1))
        np.add.at(UVGRID,num_pixels*np.ceil(vi).astype(int)+np.floor(ui).astype(int),(vi%1)*(1-ui%1))
        np.add.at(UVGRID,num_pixels*np.ceil(vi).astype(int)+np.ceil(ui).astype(int),(vi%1)*(ui%1))
    else:
        # order is 0, lets do histogram2d with numpy
        kvec -= (kvec[1]-kvec[0])/2.
        kvec = np.append(kvec,kvec[-1]+(kvec[1]-kvec[0]))
        UVGRID,edges,centers = np.histogram2d(v,u,bins=kvec)
        
    return UVGRID.reshape(num_pixels,num_pixels)

    
def generate_random_uv_config(antennaconfig=None):
    '''                                                                                                                      
    In contrast to the function from earlier, which uses a file to load a uv configuration,                                  
    this function generates a random ALMA observation from scratch.                                                          
    '''

    # random number of antennas between 30 and 50                            
    #num_ant = np.random.randint(12,51)
    num_ant = (np.floor(np.random.power(0.8)*(51-13)+13)).astype(int)
    num_baselines = num_ant*(num_ant-1)/2

    # random observing length between 10 mins and 4h, given in radians                                                      
    #obs_len = (np.random.random()*(4*3600.-600.)+600.) / (24.*3600.)*2*np.pi
#    obs_len = (np.random.random()*(12.*3600.-600.)+600.) / (24.*3600.)*2*np.pi

    # initial bad choices of declination, start time, and length
    dec = 0.
    obs_start_time = -np.pi
    obs_len=0.1

    # nested while loop to get hour angle and declination (risky)
    # require that elevation of antennas must be >30 degrees (otherwise bad psf)
    while not (is_observable(obs_start_time,dec) \
                   and is_observable(obs_start_time+obs_len,dec)):
        # Source must rise above horizon for at least 30 minutes
        while not (is_observable(-0.25/12.*np.pi,dec) \
                       and is_observable(0.25/12.*np.pi,dec)): 
            dec = np.random.random()*(np.pi-0.92)+0.92
    
        # Now lets pick an observing length and observing start time
        # random observing length between 10 mins and 4h, given in radians 
        #obs_len = (np.random.random()*(4.-1/60.)+1./60.) / (24.)*2.*np.pi
        obs_len = (np.random.power(0.5)*(4.-1/60.)+1./60.) / (24.)*2.*np.pi  # weighted to smaller times

        # Finally, a random start time (in radians).  This must be such that 
        # both the beginning and end are observable.
        obs_start_time = np.random.random()*(np.pi-obs_len)-np.pi/2. 


    # integration time is typically 6s. Lets use 60 & Convert it to radians     
    tint = 60./3600./24 *2*np.pi 
    
    # frequency is anywhere between bands 4 and 7... convert to wavelength
    nu = (np.random.random()*(350-100)+100)*10**9 
    wavelength = c / nu
    
    if antennaconfig is not None: # if antenna configuration is specified, load it.
        configdata = np.genfromtxt(antennaconfig)
        xp = configdata[:,0] / wavelength
        yp = configdata[:,1] / wavelength
        zp = configdata[:,2] / wavelength
        num_ant = len(xp)
        num_baselines  = num_ant*(num_ant-1)/2
    else:  # If no configuration specified, build new one.   
        # largest antenna distance from the origin (meters, converted to lambdas) 
        max_antenna_dist = 700.*np.random.random()+300.   # this one for training, other one is for coverage probs 
#        max_antenna_dist = 1000
        r = (np.random.random(num_ant)*(max_antenna_dist-12)+12) / wavelength
        th = np.random.random(num_ant)*2*np.pi

        # generate x,y,z coordinates (prior to projection in the uv plane)         
        xp,yp = pol2cart(r,th)
        zp = np.zeros(len(xp))
    
    xp -= np.mean(xp)
    yp -= np.mean(yp)
    zp -= np.mean(zp)

    # Hour angle of observations                                                
    HourAngle = np.arange(obs_start_time,obs_start_time+obs_len,tint)

    # Antenna indices of each baseline                                         
    inds = np.array(list(combinations(range(num_ant),2)))

    # get projected x, y, and z coordinates                                    
    x = Rotation_matrix[0,0]*xp+Rotation_matrix[0,1]*yp+Rotation_matrix[0,2]*zp
    y = Rotation_matrix[1,0]*xp+Rotation_matrix[1,1]*yp+Rotation_matrix[1,2]*zp
    z = Rotation_matrix[2,0]*xp+Rotation_matrix[2,1]*yp+Rotation_matrix[2,2]*zp
    
    ua = np.zeros([len(HourAngle),num_ant])
    va = np.zeros([len(HourAngle),num_ant])
    wa = np.zeros([len(HourAngle),num_ant])

    # get projected baseline coordinates (unnecessary?)                        
    BL_x = x[inds[:,1]]-x[inds[:,0]]
    BL_y = y[inds[:,1]]-y[inds[:,0]]
    BL_z = z[inds[:,1]]-z[inds[:,0]]

    # setup array for u and v points                                           
    u = np.zeros([len(HourAngle),num_baselines])
    v = np.zeros([len(HourAngle),num_baselines]) 
    
    
    # get uv points                                                            
    for i in range(len(HourAngle)):
        ua[i,:] , va[i,:] = xyz_to_uvw(x,y,z,HourAngle[i],dec)
        u[i,:] = ua[i,inds[:,1]] - ua[i,inds[:,0]]
        v[i,:] = va[i,inds[:,1]] - va[i,inds[:,0]]

    return u.ravel(),v.ravel()

def get_new_UVGRID_and_db(pixscale,num_pixels,antennaconfig=None,deposit_order=1):
    '''
    Generate a random uv configuration, and create a 
    corresponding UV grid (for noise adding during training)
    and an accurate dirty beam (for convolution with the image)
    '''
    u , v = generate_random_uv_config(antennaconfig)
    UVGRID = grid_deposit(u,v,pixscale,num_pixels,order=deposit_order)
    db = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(grid_deposit(u,v,pixscale,8*num_pixels)))).real[3*num_pixels:5*num_pixels,3*num_pixels:5*num_pixels]
    return UVGRID , np.fft.fftshift(np.fft.fft2(np.fft.fftshift(db)))

def xyz_to_uvw(x,y,z,ha,dec):
    
    # rotate about z axis by ha
    u = np.cos(ha) * x - np.sin(ha) * y
    v0 =np.sin(ha) * x + np.cos(ha) * y
    
    # rotate about x axis by -dec
    v = np.cos(-dec)* v0 -np.sin(-dec)*z
    w = np.sin(-dec)* v0 +np.cos(-dec)*z

    return u,v


def is_observable(ha,dec):
    '''
    check if a given hour angle and declination are observable by ALMA
    '''
    # This is the vector of ALMA's position on the unit circle
    x = np.sin(ALMA_latitude2)*np.cos(0)
    y = np.sin(ALMA_latitude2)*np.sin(0)
    z = np.cos(ALMA_latitude2)
    
    # this is the position of the source
    ha_x = np.sin(dec)*np.cos(ha)
    ha_y = np.sin(dec)*np.sin(ha)
    ha_z = np.cos(dec)
    
    dp = np.dot([x,y,z],[ha_x,ha_y,ha_z])
    
    # Observable to us means elevation is greater than 30 degrees
    # This means, dot product must be greater than 0.5
    return dp >0.5
