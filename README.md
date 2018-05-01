# DeepClean
A deep Neural Network alternative to the CLEAN algorithm for Interferometric images

Author:  Warren Morningstar

Many astronomical analyses of Interferometric images rely on some form  of method to remove the effects of the synthesized beam
(sometimes called the dirty beam).  Without going into too much detail, the dirty beam is similar to a point spread function,
except that it is finite across most of the sky (unlike typical PSFs, which are very nearly zero far away from the center).

Typical methods to deconvolve the image from the dirty beam include the CLEAN algorithm, and the maximum entropy method.  CLEAN
works by iteratively subtracting point sources convolved with the dirty beam from the image, until the residuals are consistent
with noise.  While this makes sense for highly compact sources such as quasars, this does not seem like a good idea for extended
sources such as galaxies or protoplanetary disks, as these objects are not composed of individual discrete point sources.  
While the results of this assumption do not cause CLEAN to fail <it>per se</it>, it does make the process require significant
human supervision.

In this project, I present a deep learning algorithm implementation that can perform the deconvolution task without significant 
human supervision.  At the moment, this implementation is intended to deal with (i.e. trained on) ALMA observations of 
gravitational lenses.  While it has been found to be able to generalize fairly well, it is not yet clear that it will perform
well on images that are qualitatively (or quantitatively) very different from images in the training set.  Therefore users are
cautioned.

The measurement of an image by an Interferometer can be thought of as a form of corruption of an image.  Specifically,
the true sky emission undergoes a fourier transform, and is observed at discrete points in frequency space by pairs of 
antennae in the interferometer.  Each of those observations is finite in duration, and thus receives some measurement noise.  
Therefore, while the form of corruption is known, it cannot trivially be reverse engineered.  CLEAN and maximum entropy are 
attempts to reverse engineer the underlying image using special assumptions.

The algorithm I use is an implementation of a [Recurrent Inference Machine](http://sbt.science.uva.nl/mri/author/mri/) by 
[Putzky & Welling (2017)](https://arxiv.org/abs/1706.04008).  This is a specialized form of Convolutional Recurrent Neural 
Network that is designed to solve inverse problems of the form described above.  Specifically, inverse problems for which the 
form of corruption is known, and thus a forward model can be constructed.  At each time step in the recurrent network, a 
prediction as to the true underlying image is made by the network. The inputs to the current time step are the prediction from 
the previous time step, as well as the gradient of the likelihood of that predicted image (given the observed image) with 
respect to itself.  The RIM takes these images, and (with the help of an internal memory state that is updated at each 
time step as well) produces an update to its prediction, which it adds to the prediction from the previous time step to 
produce the prediction of the current time step.  

Effectively, the RIM cell can be thought of as learning some form of prior as to the appearance of the true images, along with
an optimization scheme.  It uses the gradient of this prior, along with the gradients of the likelihood to determine an update 
to maximize the posterior probability of the predicted image.  It does all of this with relatively few parameters for a Neural 
Network (~500,000), and has been found to be able to generalize quite well (See the Putzky & Welling paper above).


Some neat images, showing the performance of the network on simulated images will appear as this repository develops.  
Hopefully, before too long we'll also make a few reconstructions of real ALMA observations.
