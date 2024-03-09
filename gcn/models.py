from .layers import *
from .metrics import *
from .layers import _LAYER_UIDS
import numpy as np


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'concat'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        self.skip = FLAGS.skip

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []
        self.utilnets = []
        self.attacts = []
        self.actor_layers = []
        self.critic_layers = []
        self.actors = []
        self.critics = []

        self.inputs = None
        self.outputs = None
        self.outputs_softmax = None
        self.outputs_utility = None
        self.pred = None
        self.output_dim = None
        self.input_dim = None
        self.actor_space = None
        self.critic_space = None

        self.loss = 0
        self.loss_crt = 0
        self.loss_act = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.optimizer = None
        self.opt_op = None
        self.optimizer_crt = None
        self.opt_op_crt = None

    def _build(self):
        raise NotImplementedError

    def _wire(self):
        raise NotImplementedError

    def _opt_set(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.compat.v1.variable_scope(self.name):
            self._build()

        self.sparse_input = tf.cast(self.inputs, dtype=tf.float32)
        self.dense_input = tf.compat.v1.sparse_tensor_dense_matmul(self.sparse_input, tf.eye(self.input_dim))
        self.normed_input = tf.ones_like(self.dense_input)
        self._wire()

        # Store model variables for easy access
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()
        self._f1()
        self._opt_set()

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _loss_reg(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def _f1(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.compat.v1.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.compat.v1.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += my_softmax_cross_entropy(self.outputs, self.placeholders['labels'])
    def _loss_reg(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        self.loss += tf.reduce_mean(tf.square(self.outputs-self.placeholders['labels']))

    def _accuracy(self):
        self.accuracy = my_accuracy(self.outputs, self.placeholders['labels'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class MLP2(Model):
    """Explicitly input hyperparameters rather than from FLAGS """
    def __init__(self, placeholders, hidden_dim,
                 act=tf.nn.leaky_relu, num_layer=1, bias=False,
                 learning_rate=0.00001, learning_decay=1.0, weight_decay=5e-4,
                 is_dual=False, is_noisy=False,
                 **kwargs):
        super(MLP2, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        self.hidden_dim = hidden_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.num_layer = num_layer
        self.placeholders = placeholders
        self.weight_decay = weight_decay
        self.learning_decay = learning_decay
        self.act = act
        self.bias = bias
        self.is_dual = is_dual

        if self.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                                initializer=tf.zeros_initializer)
            self.learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, self.global_step_tensor, 5000,
                                                                      self.learning_decay, staircase=True)
        else:
            self.learning_rate = learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs - self.placeholders['labels'])**2))
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))/tf.math.reduce_std(self.placeholders['labels'])
        mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.outputs[:, 0:self.output_dim])
        diver_loss = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        # diver_loss += self.weight_decay * tf.reduce_mean(self.outputs[:,0:self.output_dim])
        # diver_loss = tf.compat.v1.metrics.root_mean_squared_error(self.placeholders['labels'], self.outputs)
        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        # Opt 1: Plain network
        if self.is_dual:
            self.outputs = tf.reduce_mean(self.activations[-1][:, 0], 0) \
                           + (self.activations[-1][:, 1:] - tf.reduce_mean(self.activations[-1][:, 1:], 0))
        else:
            self.outputs = self.activations[-1]
        # Opt 2: Dueling network
        # self.outputs_softmax = tf.nn.softmax(self.outputs, axis=0)
        self.outputs_softmax = self.outputs
        self.outputs_utility = self.dense_input
        self.pred = tf.argmax(self.outputs)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        if self.num_layer == 1:
            self.layers.append(Dense(input_dim=self.input_dim,
                                     output_dim=self.output_dim,
                                     placeholders=self.placeholders,
                                     act=self.act,
                                     dropout=True,
                                     sparse_inputs=True,
                                     bias=self.bias,
                                     logging=self.logging))
        else:
            self.layers.append(Dense(input_dim=self.input_dim,
                                     output_dim=self.hidden_dim,
                                     placeholders=self.placeholders,
                                     act=self.act,
                                     dropout=True,
                                     sparse_inputs=True,
                                     bias=self.bias,
                                     logging=self.logging))
            for i in range(self.num_layer - 2):
                self.layers.append(Dense(input_dim=self.hidden_dim,
                                        output_dim=self.hidden_dim,
                                        placeholders=self.placeholders,
                                        act=self.act,
                                        dropout=True,
                                        bias=self.bias,
                                        logging=self.logging))

            self.layers.append(Dense(input_dim=self.hidden_dim,
                                    output_dim=self.output_dim,
                                    placeholders=self.placeholders,
                                    act=self.act,
                                    # act=lambda x: x,
                                    dropout=True,
                                    bias=self.bias,
                                    logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCN_DEEP_DIVER(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_DEEP_DIVER, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        if FLAGS.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
            learning_rate = tf.compat.v1.train.exponential_decay(FLAGS.learning_rate, self.global_step_tensor, 1000, FLAGS.learning_decay, staircase=True)
        else:
            learning_rate = FLAGS.learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        sparse_input = tf.cast(self.inputs, dtype=tf.float32)
        dense_input = tf.compat.v1.sparse_tensor_dense_matmul(sparse_input, tf.eye(self.input_dim))
        # 32 outputs
        # diver_loss = my_softmax_cross_entropy(self.outputs[:,0:self.output_dim], self.placeholders['labels'])
        diver_loss = my_weighted_softmax_cross_entropy(self.outputs[:,0:self.output_dim], self.placeholders['labels'], dense_input[:, 0])
        for i in range(1,FLAGS.diver_num):
            # diver_loss = tf.reduce_min([diver_loss, my_softmax_cross_entropy(self.outputs[:, 2*i:2*i + self.output_dim], self.placeholders['labels'])])
            diver_loss = tf.reduce_min([diver_loss, my_weighted_softmax_cross_entropy(self.outputs[:, 2 * i:2 * i + self.output_dim],
                                                                         self.placeholders['labels'], dense_input[:, 0])])
        self.loss += diver_loss

    def _loss_reg(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        self.loss += tf.reduce_mean(tf.abs(self.outputs-self.placeholders['labels']))

    def _accuracy(self):
        # 32 outputs
        acc = my_accuracy(self.outputs[:,0:self.output_dim], self.placeholders['labels'])
        for i in range(1,FLAGS.diver_num):
            acc = tf.reduce_max([acc, my_accuracy(self.outputs[:,2*i:2*i+self.output_dim], self.placeholders['labels'])])
        self.accuracy = acc

    def _f1(self):
        # 32 outputs
        f1, precision, recall = my_f1(self.outputs[:,0:self.output_dim], self.placeholders['labels'])
        for i in range(1,FLAGS.diver_num):
            f1_i, prec_i, recall_i = my_f1(self.outputs[:,2*i:2*i+self.output_dim], self.placeholders['labels'])
            f1 = tf.reduce_max([f1, f1_i])
            precision = tf.reduce_max([precision, prec_i])
            recall = tf.reduce_max([recall, recall_i])
        self.f1 = f1
        self.precision = precision
        self.recall = recall

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        if not self.skip:
            self.outputs = self.activations[-1]
        else:
            # hiddens = [dense_input] + self.activations[1:]
            hiddens = [self.dense_input] + self.activations[-1:]
            super_hidden = tf.concat(hiddens, axis=1)
            if FLAGS.wts_init == 'random':
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1])
            elif FLAGS.wts_init == 'zeros':
                input_dim = self.dense_input.get_shape().as_list()[1]
                output_dim = self.activations[-1].shape.as_list()[1]
                dense_shape = [super_hidden.get_shape().as_list()[1], output_dim]
                init_wts = np.zeros(dense_shape, dtype=np.float32)
                diag_mtx = np.identity(int(output_dim/2))
                neg_indices = list(range(0, output_dim-1, 2))
                pos_indices = list(range(1, output_dim, 2))
                init_wts[0:int(output_dim/2), neg_indices] = - diag_mtx
                init_wts[0:int(output_dim/2), pos_indices] = diag_mtx
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1], kernel_initializer=tf.constant_initializer(init_wts))

        self.outputs_softmax = tf.nn.softmax(self.outputs[:,0:2])
        for out_id in range(1, FLAGS.diver_num):
            self.outputs_softmax = tf.concat([self.outputs_softmax, tf.nn.softmax(self.outputs[:,self.output_dim*out_id:self.output_dim*(out_id+1)])], axis=1)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        for i in range(FLAGS.num_layer-2):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=tf.nn.leaky_relu,
                                                dropout=True,
                                                logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=2*FLAGS.diver_num,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs)
        return self.outputs_softmax


class GCN_DQN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_DQN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        if FLAGS.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                      initializer=tf.zeros_initializer)
            learning_rate = tf.compat.v1.train.exponential_decay(FLAGS.learning_rate, self.global_step_tensor, 5000,
                                                       FLAGS.learning_decay, staircase=True)
        else:
            learning_rate = FLAGS.learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))/tf.math.reduce_std(self.placeholders['labels'])
        # mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.outputs[:,0:self.output_dim])
        # diver_loss = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        # diver_loss = tf.compat.v1.metrics.root_mean_squared_error(self.placeholders['labels'], self.outputs[:,0:self.output_dim])

        for i in range(1, FLAGS.diver_num):
            diver_loss = tf.reduce_min([diver_loss,
                                        tf.reduce_mean(
                                            tf.abs(self.outputs[:, i:i + self.output_dim] - self.placeholders['labels']))])
        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        if not self.skip:
            self.outputs = self.activations[-1]
        else:
            # hiddens = [dense_input] + self.activations[1:]
            hiddens = [self.dense_input] + self.activations[-1:]
            super_hidden = tf.concat(hiddens, axis=1)
            if FLAGS.wts_init == 'random':
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1])
            elif FLAGS.wts_init == 'zeros':
                input_dim = self.dense_input.get_shape().as_list()[1]
                output_dim = self.activations[-1].shape.as_list()[1]
                dense_shape = [super_hidden.get_shape().as_list()[1], output_dim]
                init_wts = np.zeros(dense_shape, dtype=np.float32)
                diag_mtx = np.identity(int(output_dim/2))
                neg_indices = list(range(0, output_dim-1, 2))
                pos_indices = list(range(1, output_dim, 2))
                init_wts[0:int(output_dim/2), neg_indices] = - diag_mtx
                init_wts[0:int(output_dim/2), pos_indices] = diag_mtx
                self.outputs = tf.compat.v1.layers.dense(super_hidden, self.activations[-1].shape[1], kernel_initializer=tf.constant_initializer(init_wts))

        # self.outputs_softmax = tf.nn.softmax(self.outputs, axis=0)
        self.outputs_softmax = self.outputs
        self.outputs_utility = self.dense_input
        self.pred = tf.argmax(self.outputs)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        if FLAGS.num_layer==1:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FLAGS.diver_num,
                                                placeholders=self.placeholders,
                                                # act=tf.nn.leaky_relu,
                                                act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=True,
                                                # bias=True,
                                                logging=self.logging))
        else:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=tf.nn.leaky_relu,
                                                # act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=True,
                                                logging=self.logging))
            for i in range(FLAGS.num_layer - 2):
                self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                    output_dim=FLAGS.hidden1,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.leaky_relu,
                                                    # act=lambda x: x,
                                                    dropout=True,
                                                    logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.diver_num,
                                                placeholders=self.placeholders,
                                                # act=tf.nn.leaky_relu,
                                                act=lambda x: x,
                                                dropout=True,
                                                logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs


class GCN2_DQN(Model):
    """Explicitly input hyperparameters rather than from FLAGS """
    def __init__(self, placeholders, hidden_dim,
                 act=tf.nn.leaky_relu, num_layer=1, bias=False,
                 learning_rate=0.00001, learning_decay=1.0, weight_decay=5e-4,
                 is_dual=False, is_noisy=False,
                 **kwargs):
        super(GCN2_DQN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = placeholders['features'].get_shape().as_list()[1]
        self.hidden_dim = hidden_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.num_layer = num_layer
        self.placeholders = placeholders
        self.weight_decay = weight_decay
        self.learning_decay = learning_decay
        self.act = act
        self.bias = bias
        self.is_dual = is_dual

        if self.learning_decay < 1.0:
            self.global_step_tensor = tf.compat.v1.get_variable('global_step', trainable=False, shape=[],
                                                                initializer=tf.zeros_initializer)
            self.learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, self.global_step_tensor, 5000,
                                                                      self.learning_decay, staircase=True)
        else:
            self.learning_rate = learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        # diver_loss = tf.reduce_mean(self.placeholders['labels'])
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs - self.placeholders['labels'])**2))
        # diver_loss = tf.sqrt(tf.reduce_mean((self.outputs[:,0:self.output_dim] - self.placeholders['labels'])**2))/tf.math.reduce_std(self.placeholders['labels'])
        mse = tf.losses.mean_squared_error(self.placeholders['labels'], self.outputs[:, 0:self.output_dim])
        diver_loss = tf.sqrt(tf.reduce_mean(mse, name="loss"))
        # diver_loss += self.weight_decay * tf.reduce_mean(self.outputs[:,0:self.output_dim])
        # diver_loss = tf.compat.v1.metrics.root_mean_squared_error(self.placeholders['labels'], self.outputs)
        self.loss += diver_loss

    def _accuracy(self):
        self.accuracy = tf.constant(0, dtype=tf.float32)

    def _f1(self):
        self.f1 = tf.constant(0, dtype=tf.float32)

    def _wire(self):
        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)

        for layer in self.layers:
            if layer_id < len(self.layers)-1:
                # hidden = tf.nn.relu(layer(self.activations[-1]))
                hidden = layer(self.activations[-1]) # activation already built inside layer
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1

        # Opt 1: Plain network
        if self.is_dual:
            self.outputs = tf.reduce_mean(self.activations[-1][:, 0], 0) \
                           + (self.activations[-1][:, 1:] - tf.reduce_mean(self.activations[-1][:, 1:], 0))
        else:
            self.outputs = self.activations[-1]
        # Opt 2: Dueling network
        # self.outputs_softmax = tf.nn.softmax(self.outputs, axis=0)
        self.outputs_softmax = self.outputs
        self.outputs_utility = self.dense_input
        self.pred = tf.argmax(self.outputs)

    def _opt_set(self):
        if FLAGS.learning_decay < 1.0:
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
        else:
            # grad_vars = self.optimizer.compute_gradients(self.loss)
            # clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grad_vars]
            self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        if self.num_layer == 1:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                # act=lambda x: x,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
        else:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.hidden_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                dropout=True,
                                                sparse_inputs=True,
                                                bias=self.bias,
                                                logging=self.logging))
            for i in range(self.num_layer - 2):
                self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                    output_dim=self.hidden_dim,
                                                    placeholders=self.placeholders,
                                                    act=self.act,
                                                    dropout=True,
                                                    bias=self.bias,
                                                    logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=self.act,
                                                # act=lambda x: x,
                                                dropout=True,
                                                bias=self.bias,
                                                logging=self.logging))

        # self.layers.append(NoisyDense(input_dim=self.hidden_dim,
        #                               units=self.output_dim + 1,
        #                               logging=self.logging))

    def predict(self):
        # return tf.nn.softmax(self.outputs, axis=0)
        return self.outputs

