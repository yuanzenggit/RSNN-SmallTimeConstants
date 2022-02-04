# adaptive LIF neuron model ##########################################################
from collections import namedtuple
import numpy.random as rd
import tensorflow.compat.v1 as tf
from tensorflow.python.ops.variables import Variable
import numpy as np

#########################################################
@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0)
        dz_dv_scaled *= dampening_factor

        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="SpikeFunction"), grad

#########################################################
ALIFStateTuple = namedtuple('ALIFState', ('out','u','s','b','I','ref'))
Cell = tf.nn.rnn_cell.BasicRNNCell
class ALIF(Cell):
    def __init__(self, seed, n_in, n_rec, n_out, \
    t_ref, tau_m, tau_w, Vth, beta, tau_syn, tau_post, \
    Vmax, dampening_factor=0.3, dtype=tf.float32):       
        
        # Parameters
        self.n_in = n_in     #number of input neurons
        self.n_rec = n_rec   #number of recurrent neurons
        self.n_out = n_out
        
        self.t_ref = t_ref
        self.tau_m = tau_m
        self.tau_w = tau_w
        self.Vth = Vth
        self.beta = beta
        self.tau_syn = tau_syn
        self.tau_post = tau_post
        self.dampening_factor = dampening_factor
        self.Vmax = Vth*Vmax

        #trainable parameters:
        with tf.variable_scope('InputWeights'): #initial input weights
            rd.seed(seed)
            #self.W_in = tf.Variable(tf.ones(shape=(n_in, n_rec)), dtype=dtype, name="InputWeight")
            self.W_in = tf.Variable(rd.randn(n_in, n_rec) / np.sqrt(n_in), dtype=dtype, name="InputWeight", trainable=True)

        with tf.variable_scope('RecWeights'): #initial recurrent weights
            rd.seed(seed)
            #self.W_rec = tf.Variable(tf.zeros(shape=(n_rec, n_rec)), dtype=dtype, name='RecurrentWeight', trainable=False)
            self.W_rec_val = tf.Variable(rd.randn(n_rec, n_rec) / np.sqrt(n_rec), dtype=dtype, name='RecurrentWeight', trainable=True)
            recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
            self.W_rec = tf.where(recurrent_disconnect_mask, tf.zeros_like(self.W_rec_val),self.W_rec_val)
            #self.W_rec = self.W_rec_val
			
    @property
    def output_size(self):
        return [self.n_rec, self.n_rec, self.n_rec, self.n_rec, self.n_rec, self.n_rec, self.n_rec, self.n_rec, self.n_rec]

    @property
    def state_size(self):
        return ALIFStateTuple(out=self.n_rec, u=self.n_rec, s=self.n_rec, b=self.n_rec, I=self.n_rec, ref=self.n_rec)

    def zero_state(self, batch_size, dtype):
        n_rec = self.n_rec
        u0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype) #membrane potential
        s0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype) #spike pattern
        b0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype) #adaptive threshold 
        i0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype) #current
        ref0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype) #refractory period
        out0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype) #output potential
        return ALIFStateTuple(out=out0, u=u0, s=s0, b=b0, ref=ref0, I=i0)

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):
        with tf.name_scope('ALIFcall'):
            
            adp_decay = np.exp(-1.0 / self.tau_w)
            b = adp_decay*state.b + (1-adp_decay)*state.s      #adaptive current
            
            I1 = tf.matmul(inputs, self.W_in)                   #input current
            I2 = tf.matmul(state.s, self.W_rec)                 #recurrent current
            
            syn_decay = tf.exp(-1.0 / self.tau_syn)
            I2 = state.I*syn_decay + (1-syn_decay)*I2

            decay = tf.exp(-1.0 / self.tau_m)
            u = state.u*decay + (1-decay)*(I1+I2-self.beta*b) 
            u = tf.where(tf.greater(state.ref, 0.0), tf.zeros_like(u), u)
            ###
            u = tf.where(tf.greater(0.0, u), tf.zeros_like(u), u) #set voltage lower 
            u = tf.where(tf.greater(u, self.Vmax), tf.ones_like(u)*self.Vmax, u) #set voltage upper limit
            ###
            u_scale = (state.u-self.Vth)/self.Vth
            
            s = SpikeFunction( u_scale, self.dampening_factor)  #spike generation, gradient estimation
            s = tf.where(tf.greater(state.ref, 0.0), tf.zeros_like(s), s) #not spike in refractory period
            
            ref = tf.where(tf.greater(state.ref, 0.0), tf.ones_like(u)*(state.ref-1), state.ref) #refractory timing reduce 1 step
            ref = tf.where(tf.greater(s, 0.0), tf.ones_like(s)*self.t_ref, ref) #enter refractory period          

            # output filter
            out = state.out*tf.exp(-1.0/self.tau_post) + s*(1-tf.exp(-1.0/self.tau_post)) 
            
            new_state = ALIFStateTuple(out, u, s, b, I2, ref)
                       
        return [out, u, s, b*self.beta, ref, I1, I2, self.W_in, self.W_rec], new_state
