
# coding: utf-8

# In[1]:


import time
from collections import namedtuple

import numpy as np
import tensorflow as tf


# In[2]:


with open('kata bijak.txt', 'r') as f:
    text=f.read()

import re
words = ' '.join(re.findall(r'[\w]+',text.lower()))
words_list = words.split()
vocab = sorted(set(words.split()))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in words_list], dtype=np.int32)


# In[3]:


encoded


# In[4]:


print(vocab_to_int)


# In[5]:


print(int_to_vocab)


# In[6]:


print(text[:100])


# In[8]:


print(encoded[:100])


# In[9]:


print(len(vocab))


# In[10]:


print(len(encoded))


# In[11]:


def get_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    # Get the number of characters per batch and number of batches we can make
    characters_per_batch = n_seqs*n_steps
    n_batches = len(arr)//characters_per_batch
    
    # Keep only enough characters to make full batches
    arr = arr[:characters_per_batch*n_batches]
    
    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs,-1))
    print(arr)
    
    for n in range(0, arr.shape[1], n_steps):
        # The features
        x = arr[:, n:n+n_steps]
        # The targets, shifted by one
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y


# In[15]:


#Visualize what is happpening

# In this code n_steps is the size of horizontal width of what moves
# And n_seqs is the no. of rows 
# y is the same size as x which is n_steps just one shifted right
n_seqs = 5
n_steps = 15
arr = encoded

characters_per_batch = n_seqs*n_steps
n_batches = len(arr)//characters_per_batch

arr = arr[:characters_per_batch*n_batches]
# Reshape into n_seqs rows
arr = arr.reshape((n_seqs,-1))
print(arr.shape)
print(arr)
for n in range(0, arr.shape[1], n_steps)[:1]:
    # The features
    x = arr[:, n:n+n_steps]
    # The targets, shifted by one
    y = np.zeros_like(x)
    y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
    print("x is: ", x)
    print("y is: ", y)


# In[16]:


batches = get_batches(encoded, n_seqs, n_steps)
x, y = next(batches)


# In[17]:


print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])


# In[18]:


def build_inputs(batch_size, num_steps):
    ''' Define placeholders for inputs, targets, and dropout 
    
        Arguments
        ---------
        batch_size: Batch size, number of sequences per batch
        num_steps: Number of sequence steps in a batch
        
    '''
    # Declare placeholders we'll feed into the graph
    inputs = tf.placeholder(tf.int32, shape = [batch_size, num_steps] , name ='inputs') #10*50 for our case
    targets =  tf.placeholder(tf.int32, shape = [batch_size, num_steps] , name ='targets')
    
    # Keep probability placeholder for drop out layers
    keep_prob =  tf.placeholder(tf.float32, name ='keep_prob')
    
    return inputs, targets, keep_prob


# In[19]:


def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    ''' Build LSTM cell.
    
        Arguments
        ---------
        keep_prob: Scalar tensor (tf.placeholder) for the dropout keep probability
        lstm_size: Size of the hidden layers in the LSTM cells
        num_layers: Number of LSTM layers
        batch_size: Batch size

    '''
    ### Build the LSTM Cell
    # Use a basic LSTM cell
    def build_cell(lstm_size, keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # Add dropout to the cell outputs
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        
        return drop
    
    # Stack up multiple LSTM layers, for deep learning
    

    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    return cell, initial_state


# In[20]:


def build_output(lstm_output, in_size, out_size):
    ''' Build a softmax layer, return the softmax output and logits.
    
        Arguments
        ---------
        
        lstm_output: List of output tensors from the LSTM layer
        in_size: Size of the input tensor, for example, size of the LSTM cells
        out_size: Size of this softmax layer
    
    '''

    # Reshape output so it's a bunch of rows, one row for each step for each sequence.
    # Concatenate lstm_output over axis 1 (the columns)
    seq_output = tf.concat(lstm_output, axis =1)
    # Reshape seq_output to a 2D tensor with lstm_size columns
    x =tf.reshape(seq_output, ([-1, in_size]))
    
    # Connect the RNN outputs to a softmax layer
    with tf.variable_scope('softmax'):
        # Create the weight and bias variables here
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    
    # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
    # of rows of logit outputs, one for each step and sequence
    logits = tf.matmul(x, softmax_w) + softmax_b
    
    # Use softmax to get the probabilities for predicted characters
    out = tf.nn.softmax(logits, name = 'predictions')
    
    return out, logits


# In[21]:


def build_loss(logits, targets, lstm_size, num_classes):
    ''' Calculate the loss from the logits and the targets.
    
        Arguments
        ---------
        logits: Logits from final fully connected layer
        targets: Targets for supervised learning
        lstm_size: Number of LSTM hidden units
        num_classes: Number of classes in targets
        
    '''
    
    # One-hot encode targets and reshape to match logits, one row per sequence per step
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped =  tf.reshape(y_one_hot, (logits.get_shape()))
    
    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_reshaped, logits = logits)
    loss = tf.reduce_mean(loss)
    return loss


# In[22]:


def build_optimizer(loss, learning_rate, grad_clip):
    ''' Build optmizer for training, using gradient clipping.
    
        Arguments:
        loss: Network loss
        learning_rate: Learning rate for optimizer
    
    '''
    
    # Optimizer for training, using gradient clipping to control exploding gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer


# In[23]:


class CharRNN:
    
    def __init__(self, num_classes, batch_size=64, num_steps=50, 
                       lstm_size=128, num_layers=2, learning_rate=0.001, 
                       grad_clip=5, sampling=False):
    
        # When we're using this network for sampling later, we'll be passing in
        # one character at a time, so providing an option for that
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()
        
        # Build the input placeholder tensors
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        # Build the LSTM cell
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        ### Run the data through the RNN layers
        # First, one-hot encode the input tokens
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        
        # Run each sequence step through the RNN with tf.nn.dynamic_rnn 
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        
        # Get softmax predictions and logits
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        
        # Loss and optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)


# In[24]:


print(tf.__version__)


# In[25]:


batch_size = 10         # Sequences per batch
num_steps = 30         # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001    # Learning rate
keep_prob = 0.5         # Dropout keep probability


# In[26]:


epochs = 10
# Save every N iterations
save_every_n = 200

model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers, 
                learning_rate=learning_rate)

saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Use the line below to load a checkpoint and resume training
    saver.restore(sess, 'checkpoints/i21_l512.ckpt')
    counter = 0
    for e in range(epochs):
        # Train network
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss, 
                                                 model.final_state, 
                                                 model.optimizer], 
                                                 feed_dict=feed)
            
            end = time.time()
            print('Epoch: {}/{}... '.format(e+1, epochs),
                  'Training Step: {}... '.format(counter),
                  'Training loss: {:.4f}... '.format(batch_loss),
#                   'Training state: {:.4f}... '.format(new_state),
                  '{:.4f} sec/batch'.format((end-start)))
        
            if (counter % save_every_n == 0):
                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
    
    saver.save(sess, "./checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))


# In[27]:


def act(self, ob):
    sess = tf.get_default_session()
    print("Closed:", sess._closed)
    return sess.run([self.sample, self.vf] + [], 
        feed_dict={self.x: [ob]})


# In[28]:


print(tf.train.get_checkpoint_state('checkpoints'))


# In[29]:


def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


# In[30]:


def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    samples = [c for c in prime]
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])
        
    return ' '.join(samples)


# In[31]:


print(tf.train.latest_checkpoint('checkpoints'))


# In[32]:


checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = sample(checkpoint, 20, lstm_size, len(vocab), prime=['ayah','cinta'])
print(samp)

