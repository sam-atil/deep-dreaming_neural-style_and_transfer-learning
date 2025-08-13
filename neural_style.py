'''neural_style.py
Use Neural Style Transfer to create an image with the style of the Style Image and the content of the Content Image
Samuel Atilano
CS 343: Neural Networks
Project 4: Transfer Learning
Spring 2025
'''
import time
import tensorflow as tf
import matplotlib.pyplot as plt

import tf_util


class NeuralStyleTransfer:
    '''Runs the Neural Style Transfer algorithm on an image using a pretrained network.
    You should NOT need to import and use Numpy in this file (use TensorFlow instead).
    '''
    def __init__(self, pretrained_net, style_layer_names, content_layer_names):
        '''
        Parameters:
        -----------
        pretrained_net: TensorFlow Keras Model object. Pretrained network configured to return netAct values in
            ALL layers when presented with an input image.
        style_layer_names: Python list of str. Names of layers in `pretrained_net` that we will readout netAct values.
            These netAct values contribute the STYLE of the generated image.
        content_layer_names: Python list of str. len = 1.
            Names of layers in `pretrained_net` that we will readout netAct values. These netAct values contribute
            to the CONTENT of the generated image. We assume that this is a length 1 list (one selected layer str).
        '''
        #Instance Variables
        self.loss_history = None

        self.style_readout_model = None
        self.content_readout_model = None

        #Defining extra variables
        self.pretrained_net = pretrained_net
        self.style_layer_names = style_layer_names
        self.content_layer_names = content_layer_names

        self.num_style_layers = len(style_layer_names)
        self.num_content_layers = len(content_layer_names)


        self.initialize_readout_models(self.style_layer_names, self.content_layer_names)

    def initialize_readout_models(self, style_layer_names, content_layer_names):
        '''Creates and assigns style and content readout models to the instance variables self.style_readout_model and
        self.content_readout_model, respectively.

        Parameters:
        -----------
        style_layer_names: Python list of str. Names of layers in `pretrained_net` that we will readout netAct values.
            These netAct values contribute the STYLE of the generated image.
        content_layer_names: Python list of str. Names of layers in `pretrained_net` that we will readout netAct values.
            These netAct values contribute the CONTENT of the generated image.
        '''
        # Compute netAct values for selected layers with the style and content images
        self.style_readout_model = tf_util.make_readout_model(self.pretrained_net, style_layer_names)
        self.content_readout_model = tf_util.make_readout_model(self.pretrained_net, content_layer_names)

    def gram_matrix(self, A):
        '''Computes the Gram matrix AA^T (<-- the ^T here means transpose of the A matrix on the right).

        Parameters:
        -----------
        A: tf tensor. shape=(K, blah). Matrix of which we want to compute the Gram matrix.

        Returns:
        -----------
        The Gram matrix of A. shape=(K, K).
        '''

        return A @ tf.transpose(A)

    def style_loss_layer(self, gen_img_layer_net_acts, style_img_layer_net_acts):
        '''Computes the contribution of the current layer toward the overall style loss.

        Parameters:
        -----------
        gen_img_layer_net_acts: tf tensor. shape=(1, Iy, Ix, K).
            netActs in response to the GENERATED IMAGE input at the CURRENT STYLE layer.
        style_img_layer_net_acts: tf tensor. shape=(1, Iy, Ix, K).
            netActs in response to the STYLE IMAGE input at the CURRENT STYLE layer.

        Returns:
        -----------
        The style loss contribution for the current layer. float.
        '''
        #Instance Variables
        B, num_rows, num_cols, K = gen_img_layer_net_acts.shape.as_list()

        #Reshaping generated image and style image
        style = tf.reshape(style_img_layer_net_acts, [K, num_rows * num_cols])
        gen = tf.reshape(gen_img_layer_net_acts, [K, num_rows * num_cols])

        #Creating the Gram Matrices
        g_style = self.gram_matrix(style)
        g_gen = self.gram_matrix(gen) 

        #Style Loss Contribution
        layer_loss = tf.reduce_sum(tf.square(g_gen - g_style)) / ((2 * (K**2)) * ((num_rows * num_cols)**2))


        return layer_loss

    def style_loss(self, gen_img_net_acts, style_img_net_acts):
        '''Computes the style loss — the average of style loss contributions across selected style layers.

        Parameters:
        -----------
        gen_img_net_acts: Python list of tf tensors. len=num_style_layers.
            List of netActs in response to the GENERATED IMAGE input at the selected STYLE layers.
            Each item in the list g_layer_net_acts (a tf tensor) has shape=(1, Iy, Ix, K).
            Note that the Iy and Ix (spatial dimensions) generally differ in different layers of the network.
        style_img_net_acts: Python list of tf tensors. len=num_style_layers.
            List of netActs in response to the STYLE IMAGE input at the selected STYLE layers.
            Each item in the list gen_img_layer_net_acts (a tf tensor) has shape=(1, Iy, Ix, K).
            Note that the Iy and Ix (spatial dimensions) generally differ in different layers of the network.

        Returns:
        -----------
        The overall style loss. float.
        '''
        
        #gathering total loss across style layers
        total_loss = 0.0
        for layer in range(self.num_style_layers):
            total_loss += self.style_loss_layer(gen_img_net_acts[layer], style_img_net_acts[layer])

        #Overall Style Loss
        total_loss = total_loss / self.num_style_layers

        return total_loss

    def content_loss(self, gen_img_layer_act, content_img_layer_net_act):
        '''Computes the content loss.

        See notebook for the content loss equation.

        Parameters:
        -----------
        gen_img_layer_act: tf tensor. shape=(1, Iy, Ix, K).
            netActs in response to the GENERATED IMAGE input at the CONTENT layer.
        content_img_layer_net_act: tf tensor. shape=(1, Iy, Ix, K).
            netActs in response to the CONTENT IMAGE input at the CONTENT layer.

        Returns:
        -----------
        The content loss. float.
        '''

        #Instance Variables
        B, num_rows, num_cols, K = gen_img_layer_act.shape.as_list()

        #Content loss
        layer_loss = (tf.reduce_sum(tf.square(gen_img_layer_act - content_img_layer_net_act)))  / (2 *num_rows * num_cols * K)

        return layer_loss


    def total_loss(self, loss_style, style_wt, loss_content, content_wt):
        '''Computes the total loss by calculating the weighted sum of the style and content losses.

        Parameters:
        -----------
        loss_style: float. Style loss.
        style_wt: float. Weighting factor for the style loss toward the total.
        loss_content: float. Content loss.
        content_wt: float. Weighting factor for the content loss toward the total.

        Returns:
        -----------
        The total loss. float.
        '''
        
        #Total Loss across style and content layers
        return (style_wt * loss_style) + (content_wt * loss_content)

    def forward(self, gen_img, style_img_net_acts, content_img_net_acts, style_wt, content_wt):
        '''Performs forward pass through pretrained network with the generated image `gen_img`. In addition, computes
        the image gradients and total loss based on the SELECTED content and style layers.

        Parameters:
        -----------
        gen_img: tf tensor. shape=(1, Iy, Ix, n_chans). Generated image that is used to compute netAct values, loss,
            and the image gradients. Note that these are the raw PIXELS, NOT the netAct values.
        style_img_net_acts: Python list of tf tensors. len=num_style_layers.
            List of netActs in response to the STYLE IMAGE input at the selected STYLE layers.
            Each item in the list gen_img_layer_net_acts (a tf tensor) has shape=(1, Iy, Ix, K).
            Note that the Iy and Ix (spatial dimensions) may differ in different layers of the network.
        content_img_net_acts: tf tensor. shape=(1, Iy, Ix, K).
            netActs in response to the CONTENT IMAGE input at the CONTENT layer.
        style_wt: float. Weighting factor for the style loss toward the total.
        content_wt: float. Weighting factor for the content loss toward the total.

        Returns:
        -----------
        loss. float. Sum of the total loss.
        grads. shape=(1, Iy, Ix, n_chans). Image gradients (`dImage` aka `dloss_dImage`) — gradient of the
            generated image with respect to each of the pixels in the generated image.
        '''
        
        
        with tf.GradientTape() as tape:
            tape.watch(gen_img)

            #Conducting the Loss
            gen_style_acts = self.style_readout_model(gen_img)
            gen_content_acts = self.content_readout_model(gen_img)

            style_loss = self.style_loss(gen_style_acts, style_img_net_acts)
            content_loss = self.content_loss(gen_content_acts, content_img_net_acts)

            total_loss = self.total_loss(style_loss, style_wt, content_loss, content_wt)

        #Conducting the Image Gradient
        d_img = tape.gradient(total_loss, gen_img)

        return total_loss.numpy(), d_img



    def fit(self, gen_img, style_img, content_img, n_epochs=200, style_wt=1e2, content_wt=1, lr=0.01,
            print_every=25, plot=True, plot_fig_sz=(5, 5), export=True):
        '''run Neural Style Transfer on the generated image by 
           modifying the generated image with the image gradient for n_epochs.

        Parameters:
        -----------
        gen_img: tf tensor. shape=(1, Iy, Ix, n_chans). Generated image that will be modified across epochs.
            This is a trainable tf tensor (i.e. tf.Variable)
        style_img: tf tensor. shape=(1, Iy, Ix, n_chans). Style image. Used to derive the style that is applied to the
            generated image. This is a constant tf tensor.
        content_img: tf tensor. shape=(1, Iy, Ix, n_chans). Content image. Used to derive the content that is applied to
            generated image. This is a constant tf tensor.
        n_epochs: int. Number of epochs to run neural style transfer on the generated image.
        style_wt: float. Weighting factor for the style loss toward the total loss.
        content_wt: float. Weighting factor for the content loss toward the total loss.
        lr: float. Learning rate.
        print_every: int. Print out progress (current epoch) every this many epochs.
        plot: bool. If true, plot/show the generated image `print_every` epochs.
        plot_fig_sz: tuple of ints. The plot figure size (height, width) to use when plotting/showing the generated image.
        export: bool. Whether to export a JPG image to the `neural_style_output` folder in the working directory
            every `print_every` epochs. Each exported image should have the current epoch number in the filename so that
            the image currently exported image doesn't overwrite the previous one.

        Returns:
        -----------
        self.loss_history. Python list of float. Loss values computed on every epoch of training.
        '''

        #Empty Loss History list
        self.loss_history = []


        #Obtaining the net_acts from the style and content images
        style_net_acts = self.style_readout_model(style_img)
        content_net_acts = self.content_readout_model(content_img)

        #Applying the Gradient with Adam Optimizer
        adam = tf.optimizers.Adam(learning_rate = lr)

        epoch_count = 0
        img_count = 1
        for epoch in range(n_epochs):
            #Conducting the forward pass
            loss, d_img = self.forward(gen_img, style_net_acts, content_net_acts, style_wt, content_wt)

            adam.apply_gradients([(d_img, gen_img)])
            gen_img.assign(tf.clip_by_value(gen_img, 0.0, 1.0))

            #Boolean Statements
            if epoch_count % print_every == 0:
                print(f'Epoch:{epoch_count}/{n_epochs}. Training Loss: {loss}')
                if plot:
                    #converting to img
                    img = tf_util.tf2image(gen_img)
                    img.show()
                if export:
                    #Exporting img
                    img.save(f'deep_dream_output/image_{img_count}.jpg', 'JPEG')
                    img_count += 1

            self.loss_history.append(loss)
            epoch_count += 1


        return self.loss_history