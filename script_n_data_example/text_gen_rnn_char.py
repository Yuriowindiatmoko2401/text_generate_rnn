
# coding: utf-8

# In[1]:


import time
from collections import namedtuple
import numpy as np


# In[31]:


import pandas as pd

tabel_1 = pd.read_csv('kata2bijak1_1.txt', sep="|",
                         engine='python', header=None, skiprows=1, 
                         names=["id","text","hash1","hash2","hash3","hash4"])

teks_only= [i for i in tabel_1['text']]
gabung = ' '.join(teks_only)
gabung

def clean_str_1(s):
    s = re.sub('http[s]?:\/\/[\w\/\.]+',' ',s)
    s = re.sub('pic\.twitter\.com/[a-zA-Z0-9]+',' ',s)
    s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)
    s = re.sub(r" : ", ":", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"pic", " ", s)
    s = re.sub(r"com", " ", s)
    s = re.sub(r"twitter", " ", s)
    s = re.sub(r"\\", " ", s)
    return s.strip()
text = clean_str_1(gabung)

vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)


# In[40]:


print(len(encoded))

print(vocab_to_int)

print(int_to_vocab)

print(text[:100])

print(encoded[:100])

print(len(vocab))


# In[41]:


def get_batches(arr, n_seqs, n_steps):

    characters_per_batch = n_seqs*n_steps
    n_batches = len(arr)//characters_per_batch

    arr = arr[:characters_per_batch*n_batches]

    arr = arr.reshape((n_seqs,-1))
    print(arr)
    
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n+n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y


# In[42]:


n_seqs = 10
n_steps = 50
arr = encoded

characters_per_batch = n_seqs*n_steps
n_batches = len(arr)//characters_per_batch

arr = arr[:characters_per_batch*n_batches]

arr = arr.reshape((n_seqs,-1))
print(arr.shape)
print(arr)
for n in range(0, arr.shape[1], n_steps)[:1]:

    x = arr[:, n:n+n_steps]

    y = np.zeros_like(x)
    y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
    print("x is: ", x)
    print("y is: ", y)


# In[48]:


batches = get_batches(encoded, 10, 50)
x, y = next(batches)


# In[50]:


print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])


# In[51]:


def build_inputs(batch_size, num_steps):

    inputs = tf.placeholder(tf.int32, shape = [batch_size, num_steps] , name ='inputs') #10*50 for our case
    targets =  tf.placeholder(tf.int32, shape = [batch_size, num_steps] , name ='targets')

    keep_prob =  tf.placeholder(tf.float32, name ='keep_prob')
    
    return inputs, targets, keep_prob


# In[52]:


def build_lstm(lstm_size, num_layers, batch_size, keep_prob):

    def build_cell(lstm_size, keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        
        return drop
  
    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    return cell, initial_state


# In[53]:


def build_output(lstm_output, in_size, out_size):

    seq_output = tf.concat(lstm_output, axis =1)

    x =tf.reshape(seq_output, ([-1, in_size]))

    with tf.variable_scope('softmax'):

        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))

    logits = tf.matmul(x, softmax_w) + softmax_b

    out = tf.nn.softmax(logits, name = 'predictions')
    
    return out, logits


# In[54]:


def build_loss(logits, targets, lstm_size, num_classes):

    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped =  tf.reshape(y_one_hot, (logits.get_shape()))

    loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_reshaped, logits = logits)
    loss = tf.reduce_mean(loss)
    return loss


# In[55]:


def build_optimizer(loss, learning_rate, grad_clip):

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer


# In[56]:


class CharRNN:
    
    def __init__(self, num_classes, batch_size=64, num_steps=50, 
                       lstm_size=128, num_layers=2, learning_rate=0.001, 
                       grad_clip=5, sampling=False):

        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()
 
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        x_one_hot = tf.one_hot(self.inputs, num_classes)
 
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state

        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
 
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)


# In[57]:


import tensorflow as tf

print(tf.__version__)


# In[58]:


batch_size = 100        # Sequences per batch
num_steps = 100          # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001    # Learning rate
keep_prob = 0.5         # Dropout keep probability


# In[59]:


epochs = 300
# Save every N iterations
save_every_n = 200

model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers, 
                learning_rate=learning_rate)

saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Use the line below to load a checkpoint and resume training
#     saver.restore(sess, 'checkpoint_kata2bijak1/i900_l512.ckpt')
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
                saver.save(sess, "checkpoint_kata2bijak1/i{}_l{}.ckpt".format(counter, lstm_size))
    
    saver.save(sess, "./checkpoint_kata2bijak1/i{}_l{}.ckpt".format(counter, lstm_size))


# In[60]:


def act(self, ob):
    sess = tf.get_default_session()
    print("Closed:", sess._closed)
    return sess.run([self.sample, self.vf] + [], 
        feed_dict={self.x: [ob]})


# In[61]:


tf.train.get_checkpoint_state('checkpoint_kata2bijak1')


# In[62]:


def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


# In[63]:


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
        
    return ''.join(samples)


# In[64]:


tf.train.latest_checkpoint('checkpoint_kata2bijak1')


# In[65]:


checkpoint = tf.train.latest_checkpoint('checkpoint_kata2bijak1')
samp = sample(checkpoint, 280, lstm_size, len(vocab), prime="kantor")
print(samp)

