# DeepClean
A deep Neural Network alternative to the CLEAN algorithm for Interferometric images

Author:  Warren Morningstar

Interferometers, such as ALMA, measure the fourier transform of the sky.  By inverting this transformation, we are able to make images at unbelievable angular resolution ( approaching milliarcseconds for ALMA or micro-arcseconds for the Event Horizon Telescope).  This angular resolution comes at a significant cost however.  A direct inverse transform will result in a poor quality image.  A few examples of (simulated) ALMA images may be seen below.

<p align="center"> 
<img src="https://github.com/wmorning/DeepClean/blob/master/images/ALMAims.png">
</p>

Some of the texture is caused by noise, which can be reduced as longer observations are taken.  However, a significant portion occurs simply due to the interferometric observing process.  The incomplete sub-sampling of the fourier transform introduces a smearing analogous to a point spread function from optical astronomy.  We call this the synthesized beam (or sometimes the dirty beam).  Depending on how your interferometer is set up, how long it observes, and where it points on the sky, the synthesized beam can be radically different, leading to the same image posessing significant variety, as shown below.

<p align="center"> 
<img src="https://github.com/wmorning/DeepClean/blob/master/images/ALMAims2.png">
</p>

Many astronomical analyses of Interferometric images rely on some form  of method to remove the synthesized beam.   The most commonly used method for this is the CLEAN algorithm. CLEAN works by iteratively subtracting point sources convolved with the dirty beam from the image, until a user-defined stopping criterion is met.  This can be a problem for a number of reasons:  1)  It can require active supervision to achieve acceptable results.  2)  It makes erroneous assumptions about the structure of the source.  3)  It can be difficult to perform without a substantial amount of practice.

<p align="center"> 
<img src="https://github.com/wmorning/DeepClean/blob/master/images/CLEAN_sequence.png">

Above is an example of CLEAN in practice.  For this, I used real ALMA observations of a gravitational lens, and personally supervised the whole process (I should note that I am a novice to CLEAN, and better performance may be found if an expert were to do this instead of me).  CLEAN works as follows:  

1.  You observe an ALMA image (shown on the left).  
2.  You forward model the image by iteratively subtracting point sources convolved with the dirty beam.  This gives you the image shown in the fourth column as your predicted image.   The image in the second column is the forward model.  
3.  You continue to do this until the error is low (shown in the third column).  Note that while there is still texture to the error (it is nonzero where the source was), that error is small.  
4.  You then choose a so-called "Clean beam" and smooth the image.  This gives you the image in the fifth column.  
5.  Finally, you then add back in the residuals, giving you the "clean image", which is shown on the right.
</p>


In this project, I present a DeepClean:  a deep learning algorithm that can perform the deconvolution task without  human supervision.  At the moment, this implementation is intended to deal with (i.e. trained on) ALMA observations of  gravitational lenses.  While we have found it be able to generalize fairly well, it is not yet clear how well it will  perform on images that are qualitatively very different from images in the training set.

The measurement of an image by an Interferometer can be thought of as a form of corruption of an image.  Specifically, the true sky emission undergoes a fourier transform, and is observed at discrete points in frequency space by pairs of  antennae in the interferometer.  Each of those observations is finite in duration, and thus receives some measurement noise.   Therefore, while the form of corruption is known, it cannot trivially be reverse engineered.  CLEAN and maximum entropy are  attempts to reverse engineer the underlying image using special assumptions.

The network I use is a [Recurrent Inference Machine](http://sbt.science.uva.nl/mri/author/mri/) by  [Putzky & Welling (2017)](https://arxiv.org/abs/1706.04008).  This is a specialized form of Convolutional Recurrent Neural  Network that is designed to solve inverse problems of the form described above.  Specifically, inverse problems for which the  form of corruption is known, and thus a forward model can be constructed.  At each time step in the recurrent network, a  prediction as to the true underlying image is made by the network. The inputs to the current time step are the prediction from  the previous time step, as well as the gradient of the likelihood of that predicted image (given the observed image) with  respect to itself.  The RIM takes these images, and (with the help of an internal memory state that is updated at each  time step as well) produces an update to its prediction, which it adds to the prediction from the previous time step to  produce the prediction of the current time step.  A diagram of the RIM is shown below:

<p align="center"> 
<img src="https://github.com/wmorning/DeepClean/blob/master/images/RIM_diagram2.png" width="50%">
</p>

the image x&#770;<sub>t-1</sub> is given to the network, along with the gradient of the log-likelihood of the data given x&#770;<sub>t-1</sub>.  This network then produces an update rule &Delta;x&#770;<sub>t</sub> which is used to get x&#770;<sub>t</sub>.  It also updates a hidden state h<sub>t</sub> which is used by the CRNN to allow it to exhibit more dynamic temporal behavior.  After a number of time steps, the output image x&#770;<sub>t</sub> is taken to be the deconvolved image.  We train by minimizing the squared error of this output image.

<p align="center"> 
<img src="https://github.com/wmorning/DeepClean/blob/master/images/RIM_time_sequence.png" >
</p>

Effectively, the RIM cell can be thought of as learning some form of prior as to the appearance of the true images, along with an optimization scheme.  It uses the gradient of this prior, along with the gradients of the likelihood to determine an update  to maximize the posterior probability of the predicted image.  It does all of this with relatively few parameters for a Neural  Network (~500,000 for theirs.  Ours uses more), and has been found to be able to generalize quite well (See the Putzky &  Welling paper above).

## DeepClean in Action

Shown below is an example of DeepClean in action.  ALMA has observed a number of gravitational lenses to date, one of 
which is the gravitational lens SPT 0529.  Using the visibilities observed from this system, and by training on a set of
hundreds of thousands of simulated (and only simulated) lenses, the RIM has learned the following procedure to perform a
reconstruction.

<p align="center"> 
<img src="https://github.com/wmorning/DeepClean/blob/master/images/SPT0529_realtime.gif">
</p>

What we see is that it initially identifies the most probable locations of emission and adds flux to those locations.  It then
adjusts its model to identify dimmer sources and to correct the morphology of the bright clumps.  The end result is a 
substantially cleaned version of the image.

This is all good for reconstructing images of lenses that have 0.04 arcsecond pixel sizes, and are 192 by 192 pixels.  
However, most sources in the universe are not lenses, and many ALMA configurations do not really observe at this exact range 
of resolutions.  The question that I'm leading to here is:  How does this network perform on observations that it was not 
originally designed to analyze?  This is obviously an valid question, since traditionally machine learning methods perform 
substantially worse outside of the distribution of data covered by their training set.  We wanted to test this (slightly), so 
what we've done is to make a synthetic ALMA observation of a non-gravitationally lensed nearby galaxy.  For this, we use the 
spiral galaxy M51, where the image for the source was found [here](https://casaguides.nrao.edu/index.php/Simalma_(CASA_4.1)).
If you download this image, you'll notice that it is both not a gravitational lens, and not 192x192 pixels.  So it is well 
outside of the training data distribution.  We then created visibilities and fed them to the network.  The reconstruction is 
shown below.

<p align="center"> 
<img src="https://github.com/wmorning/DeepClean/blob/master/images/CLEAN_generalization.png">
</p>

Did it do perfect?  Absolutely not.  But it did much better than random guessing.  It recovers the spiral structure for 
example.  That is pretty amazing, given that this network has never seen a single image of a non-lensed galaxy, nor one with 
so many pixels and such compact and complex structure.  Perhaps even more importantly, it does not spuriously add structure 
everywhere.  We will continue to study this potential, as well as trying to utilize a more comprehensive training set, so that 
we do not limit ourselves to images of gravitational lenses.  Stay tuned!
