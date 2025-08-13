'''deep_dream.py
Generate art with a pretrained neural network using the DeepDream algorithm
Samuel Atilano
CS 343: Neural Networks
Project 4: Transfer Learning
Spring 2025
'''
import time
import tensorflow as tf
import matplotlib.pyplot as plt

import tf_util


class DeepDream:
    '''Runs the DeepDream algorithm on an image using a pretrained network.
    You should NOT need to import and use Numpy in this file (use TensorFlow instead).
    '''
    def __init__(self, pretrained_net, selected_layers_names):
        '''DeepDream constructor.

        Parameters:
        -----------
        pretrained_net: TensorFlow Keras Model object. Pretrained network configured to return netAct values in
            ALL layers when presented with an input image.
        selected_layers_names: Python list of str. Names of layers in `pretrained_net` that we will readout netAct values
            from in order to contribute to the generated image.
        '''

        #Global Variables
        self.loss_history = None

        self.pretrained_net = pretrained_net
        self.selected_layers_names = selected_layers_names
        self.readout_model = tf_util.make_readout_model(pretrained_net, selected_layers_names)


    def loss_layer(self, layer_net_acts):
        '''Computes the contribution to the total loss from the current layer with netAct values `layer_net_acts`. The
        loss contribution is the mean of all the netAct values in the current layer.

        Parameters:
        -----------
        layer_net_acts: tf tensor. shape=(1, Iy, Ix, K). The netAct values in the current selected layer. K is the
            number of kernels in the layer.

        Returns:
        -----------
        loss component from current layer. float. Mean of all the netAct values in the current layer.
        '''

        #Current Layer's Loss
        loss = tf.reduce_mean(layer_net_acts)
        return loss

    def forward(self, gen_img, standardize_grads=True, eps=1e-8):
        '''Performs forward pass through the pretrained network with the generated image.
        Loss is calculated based on the SELECTED layers from the readout model.

        Parameters:
        -----------
        gen_img: tf tensor. shape=(1, Iy, Ix, n_chans). Generated image that is used to compute netAct values, loss,
            and the image gradients. The singleton dimension is the batch dimension (N).
        standardize_grads: bool. Should we standardize the image gradients?
        eps: float. Small number used in standardization to prevent possible division by 0 (i.e. if stdev is 0).

        Returns:
        -----------
        loss. float. Sum of the loss components from all the selected layers.
        grads. shape=(1, Iy, Ix, n_chans). Image gradients (`dImage` aka `dloss_dImage`) — gradient of the
            generated image with respect to each of the pixels in the generated image.

        '''

        with tf.GradientTape() as tape:
            tape.watch(gen_img)

            #Conducting the Loss
            net_acts = self.readout_model(gen_img)

            loss = []
            for net_act in net_acts:
                loss.append(self.loss_layer(net_act))

            total_loss = self.loss_layer(loss)

        #Conducting the Image Gradient
        d_img = tape.gradient(total_loss, gen_img)

        if standardize_grads:
            d_img = (d_img - tf.reduce_mean(d_img)) / (tf.math.reduce_std(d_img) + eps)

        return total_loss, d_img


    def fit(self, gen_img, n_epochs=26, lr=0.01, print_every=25, plot=True, plot_fig_sz=(5, 5), export=True):
        '''In other words, run DeepDream on the generated image by modifying it with image gradients for n_epochs 
        utilizing the ASCENT algorithm.

        Parameters:
        -----------
        gen_img: tf tensor. shape=(1, Iy, Ix, n_chans). Generated image that is used to compute netAct values, loss,
            and the image gradients.
        n_epochs: int. Number of epochs to run gradient ascent on the generated image.
        lr: float. Learning rate.
        print_every: int. Print out progress (current epoch) every this many epochs.
        plot: bool. If true, plot/show the generated image `print_every` epochs.
        plot_fig_sz: tuple of ints. The plot figure size (height, width) to use when plotting/showing the generated image.
        export: bool. Whether to export a JPG image to the `deep_dream_output` folder in the working directory
            every `print_every` epochs. Each exported image should have the current epoch number in the filename so that
            the image currently exported image doesn't overwrite the previous one. For example, image_1.jpg, image_2.jpg,
            etc.

        Returns:
        -----------
        self.loss_history. Python list of float. Loss values computed on every epoch of training.
        '''
        
        loss_history = []
        epoch_count = 0
        img_count = 1
        for epoch in range(n_epochs):
            #Conducting the forward pass
            loss, d_img = self.forward(gen_img)

            #Applying the Gradient Ascent update
            gen_img.assign_add(lr * d_img)
            gen_img.assign(tf.clip_by_value(gen_img, 0.0, 1.0))

            #Boolean Statements
            if epoch_count % print_every == 0:
                print(f'Epoch:{epoch_count}/{n_epochs}. Training Loss: {loss}')
                if plot:
                    #converting to img
                    img = tf_util.tf2image(gen_img)
                    img.show()
                if export:
                    #Exporting Image
                    img.save(f'deep_dream_output/image_{img_count}.jpg', 'JPEG')
                    img_count += 1

            loss_history.append(loss)
            epoch_count += 1

        self.loss_history = loss_history

        return loss_history


    def fit_multiscale(self, gen_img, n_scales=4, scale_factor=1.3, n_epochs=26, lr=0.01, print_every=1, plot=True,
                       plot_fig_sz=(5, 5), export=True):
        '''Run DeepDream `fit` on the generated image `gen_img` a total of `n_scales` times. After each time, scale the
        width and height of the generated image by a factor of `scale_factor` (round to nearest whole number of pixels).
        The generated image does NOT start out from scratch / the original image after each resizing. Any modifications
        DO carry over across runs.

        Parameters:
        -----------
        gen_img: tf tensor. shape=(1, Iy, Ix, n_chans). Generated image that is used to compute netAct values, loss,
            and the image gradients.
        n_scales: int. Number of times the generated image should be resized and DeepDream should be run.
        n_epochs: int. Number of epochs to run gradient ascent on the generated image.
        lr: float. Learning rate.
        print_every: int. Print out progress (current scale) every this many SCALES (not epochs).
        plot: bool. If true, plot/show the generated image `print_every` SCALES.
        plot_fig_sz: tuple of ints. The plot figure size (height, width) to use when plotting/showing the generated image.
        export: bool. Whether to export a JPG image to the `deep_dream_output` folder in the working directory
            every `print_every` SCALES. Each exported image should have the current scale number in the filename so that
            the image currently exported image doesn't overwrite the previous one.

        Returns:
        -----------
        self.loss_history. Python list of float. Loss values computed on every epoch of training.
        '''
        
        #Instance Variables
        singleton, height, width, n_chans = gen_img.shape
        loss_history = []
        scale_count = 0 
        img_count = 0    

      
        for n in range(n_scales):
            loss_his = self.fit(gen_img, n_epochs = n_epochs, lr = lr, plot = False, export = False)

            loss_history.extend(loss_his)

            #Scaling
            height = int(round(height * scale_factor))
            width = int(round(width * scale_factor))
          
            gen_img = tf.Variable(tf.image.resize(gen_img, size = (height, width)))

            #Boolean Statements
            if scale_count % print_every == 0:
                print(f'Scale:{scale_count}/{n_scales}.')
                if plot:
                    #converting to img
                    img = tf_util.tf2image(gen_img)
                    img.show()
                if export:
                    #Exporting Img
                    img.save(f'deep_dream_output/image_{img_count}.jpg', 'JPEG')
                    img_count += 1
            scale_count += 1

        self.loss_history = loss_history
        return loss_history
