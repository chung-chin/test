import tensorflow as tf

class conv1dTP(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 output_padding=None,
                 data_format='channels_last',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None, **kwargs):
        super(conv1dTP,self).__init__(trainable=trainable, name=name,
        activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
                                                                   **kwargs)
        self.filters     = filters
        self.kernel_size = (kernel_size,)
        self.strides     = (strides,)
        self.padding     = padding

        if(self.padding == 'causal'): raise ValueError('Causal is not allowed.')

        self.output_padding = output_padding
        
        if self.output_padding is not None:
            self.output_padding = (self.output_padding,)

            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) + ' must be '
                    'greater than output padding ' + str(self.output_padding))
          
        self.data_format = data_format
        self.dilation_rate = (dilation_rate,)
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.input_spec = tf.keras.layers.InputSpec( ndim = 3 )
        self.filters = filters

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Inputs should have rank 3.',
                             ' Received input shape: ' + str(input_shape))

        channel_axis = self._get_channel_axis()

        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = int(input_shape[channel_axis])
        self.input_spec = tf.keras.layers.InputSpec(ndim=3,
                            axes={channel_axis: input_dim})
        kernel_shape = self.kernel_size + (self.filters, input_dim)
        
        self.kernel = self.add_weight( name = 'kernel', shape = kernel_shape,
                initializer = self.kernel_initializer,
                regularizer = self.kernel_regularizer,
                constraint  = self.kernel_constraint,
                trainable   = True, dtype = self.dtype)
        if self.use_bias:
            self.bias = self.add_weight( name = 'bias', shape =(self.filters,),
                initializer = self.bias_initializer,
                regularizer = self.bias_regularizer,
                constraint  = self.bias_constraint,
                trainable   = True, dtype = self.dtype)
        else:
            self.bias = None
            
        self.built = True

    def call(self, inputs):
        input_shape = inputs.get_shape()
        batch_size = input_shape[0]

        l_axis   = - self._get_channel_axis()
        length   = input_shape[l_axis]
        kernel_l = self.kernel_size
        stride_l = self.strides

        if self.output_padding is None: out_pad_l = None
        else:                           out_pad_l = self.output_padding

        # Infer the dynamic output shape:
        out_length = self.deconv_output_length(length, kernel_l[0],
                     padding=self.padding, output_padding=out_pad_l,
                     stride=stride_l[0], dilation=self.dilation_rate[0])
        
        if self.data_format == 'channels_first':
              output_shape = (batch_size, self.filters, out_length  )
        else: output_shape = (batch_size, out_length  , self.filters)

        output_shape_tensor = tf.stack(output_shape)
        DataFormat = self.convert_data_format(self.data_format, ndim=3)

        PADDING = self.padding.upper()
        outputs = tf.nn.conv1d_transpose( inputs, self.kernel,
        output_shape_tensor, strides=self.strides, padding=PADDING,
        data_format=DataFormat, dilations=self.dilation_rate)

        if self.use_bias:
            outputs = tf.nn.bias_add( outputs, self.bias, data_format=DataFormat)

        if self.activation is not None: return self.activation(outputs)
        return outputs
    
    def _get_channel_axis(self):
        if self.data_format == 'channels_first': return   1
        else:                                    return - 1

    def convert_data_format(self, data_format, ndim):
        if   data_format == 'channels_last' : return 'NWC'
        elif data_format == 'channels_first': return 'NCW'
        else: raise ValueError('Invalid data_format:', data_format)
        
    def deconv_output_length(self, input_length, filter_size, padding,
                         output_padding=None, stride=0, dilation=1):
        assert padding in {'same', 'valid', 'full'}

        if input_length is None: return None

        # Get the dilated kernel size
        filter_size = filter_size + ( filter_size - 1 ) * ( dilation - 1 )

        # Infer length if output padding is None, else compute the exact length
        if output_padding is None:
            if padding == 'valid':
                length = input_length * stride + max(filter_size - stride, 0)
            elif padding == 'full':
                length = input_length * stride - (stride + filter_size - 2)
            elif padding == 'same':
                length = input_length * stride
        else:
            if   padding == 'same' : pad = filter_size // 2
            elif padding == 'valid': pad = 0
            elif padding == 'full' : pad = filter_size - 1

            length = ((input_length - 1) * stride + filter_size - 
                                       2 * pad + output_padding[0])
        return length

