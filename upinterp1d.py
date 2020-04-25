import tensorflow as tf

class upinterp1d(tf.keras.layers.Layer):
    def __init__(self,
                 size=2,
                 data_format='channels_last',
                 interpolation='cubic', **kwargs):
        super(upinterp1d, self).__init__(**kwargs)
        self.s           = int(size)
        self.data_format = data_format
        self.interp      = interpolation
        self.input_spec  = tf.keras.layers.InputSpec( ndim = 3 )

    @tf.function
    def call(self, inputs):
        order = 4
        
        if   self.interp == "linear": order, m2d = 2, 'bilinear'
        elif self.interp ==  'cubic':        m2d =    'bicubic'
        elif self.interp == 'gaussian' or self.interp == 'lanczos3' or \
             self.interp == 'lanczos5' or self.interp == 'nearest'  or \
             self.interp == 'mitchellcubic': m2d = self.interp
        else: raise ValueError("Interpolation argument is wrong in upinterp1d.")
        
        inputt = tf.stack([inputs for _ in range(order)])
        
        if   self.data_format == 'channels_last' :
            inputt = tf.transpose(inputt, perm=[1,2,0,3])
        elif self.data_format == 'channels_first':
            inputt = tf.transpose(inputt, perm=[1,3,0,2])
        else: raise ValueError('data_form argument is not right in upinterp1d.')
  
        output = tf.image.resize(inputt, method=m2d,
                 size=[inputt.shape[1] * self.s, inputt.shape[2]])
        output = tf.convert_to_tensor(output[:,:,0,:])
        
        if self.data_format == 'channels_first':
            output = tf.transpose(output, perm=[0,2,1])
            
        if inputs.dtype.is_integer: output = tf.round(output)        
        return tf.cast(output, dtype=inputs.dtype)
    
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()

        if   self.data_format == 'channels_last' :
            length = self.size * input_shape[1] if input_shape[1] is not None else None
            return tf.TensorShape([input_shape[0], length, input_shape[2]])
        elif self.data_format == 'channels_first':
            length = self.size * input_shape[2] if input_shape[2] is not None else None
            return tf.TensorShape([input_shape[0], input_shape[1], length])
        else: raise ValueError('data_form argument is not right in upinterp1d.')

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format,
                  'interpolation': self.interp}
        base_config = super(upinterp1d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    print("You ran this module directly.")
    input("\n\nPress the enter key to exit.")
