from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LOG_DIR = 'processed'
VOCAB_FILE = 'vocab.tsv'
MODEL_NAME = 'model.ckpt'

def read_data(file_path):
    print('Reading data...')
    with open(file_path, 'r') as f:
        words = f.readline().split()
    return words    
    
def build_vocab(words, vocab_size):
    print('Building vocabulary...')
    vocab = ['UNK']
    vocab.extend([word for word, _ in Counter(words).most_common(vocab_size-1)])
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    with open(os.path.join(LOG_DIR, VOCAB_FILE), 'w') as f:
        f.write('\n'.join(vocab))
    return {word: i for i, word in enumerate(vocab)}
    
def words_to_indicies(words, index):
    print('Converting words to indicies...')
    return [index[w] if w in index else 0 for w in words]
    
def get_pairs(words, window_size):
    print('Getting word pairs...')
    for i, center in enumerate(words):
        targets = words[max(0, i - window_size):i]
        targets.extend(words[i+1:i+window_size+1])
        for t in targets:
            yield center, t
            
file_path = '/media/wangx/HDD1/stanford-tensorflow-tutorials/examples/data/text8'
vocab_size = 5000
window_size = 5
embed_size = 300

words = read_data(file_path)
index = build_vocab(words, vocab_size)
index_words = words_to_indicies(words, index)

print('Building co-occurence matrix...')
occurence = np.zeros([vocab_size, vocab_size])
for center, target in get_pairs(index_words, window_size):
    occurence[center][target] += 1
    
print('Building and running graph...')
mean_occurence = tf.reduce_mean(occurence, axis=1, keep_dims=True)
mean_centered_occurence = tf.subtract(occurence, mean_occurence)
svd = tf.svd(mean_centered_occurence)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    s, u, _ = sess.run(svd)
    
    embedding_var = tf.Variable(np.dot(u, np.diag(s)[:, :embed_size]), name='embedding')
    sess.run(embedding_var.initializer)
    
    config = projector.ProjectorConfig()
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = VOCAB_FILE
    
    projector.visualize_embeddings(summary_writer, config)
    saver_embed = tf.train.Saver([embedding_var])
    saver_embed.save(sess, os.path.join(LOG_DIR, MODEL_NAME))


            
