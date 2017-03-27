class FullMatch(Layer):
    def __init__(self, output_dim=1, **kwargs):
        self.output_dim = output_dim
        self.strategy = strategy
        super(VanillaCosine, self).__init__(**kwargs)

    def build(self, input_shape):
        self.sentence_length = input_shape[0][1]
        self.built = True

    def call(self, x, mask=None):
        assert type(x) is list, "tensor is not a list"
        p, q = x # p, q :: (batch, sentence_length, 100)

        word_distances = []
        for word_index in range(self.sentence_length):
            di = cosine_distance(p[:,word_index,:], q[:,-1,:]) # (32, 1)
            word_distances.append(di)
        result = K.stack(word_distances) # (10, 32, 1) => actually want (32,1,10), this is done below
        return tf.transpose(result,(1,2,0))


    def get_output_shape_for(self, input_shape):
        shape1, shape2 = input_shape
        return (shape1[0], self.output_dim, shape1[1])

class MaxPoolingMatch(Layer):
    def __init__(self, output_dim=1, **kwargs):
        self.output_dim = output_dim
        self.strategy = strategy
        super(VanillaCosine, self).__init__(**kwargs)

    def build(self, input_shape):
        self.sentence_length = input_shape[0][1]
        self.built = True

    def call(self, x, mask=None):
        assert type(x) is list, "tensor is not a list"
        p, q = x # p, q :: (batch, sentence_length, 100)

        word_distances = []
        for word_index in range(self.sentence_length): # i :: word in p
            dij = cosine_distance(p[:,word_index,:], q[:,:,:]) # (32, 10)
            word_distances.append(dij) # 10*10 matches
        (100, 32, 1) => (32, 1, 100)
        result = K.stack(word_distances) # (10, 32, 1) => actually want (32,1,10), this is done below
        return tf.transpose(result,(1,2,0))


    def get_output_shape_for(self, input_shape):
        shape1, shape2 = input_shape
        return (shape1[0], self.output_dim, shape1[1])
