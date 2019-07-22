#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This script performs the main language-modeling experiments for the paper "Agglomerative Attention" by Matthew Spellings. For a plaintext description of the experiments, please see the paper.
# 
# ## `keras_transformer` Installation
# 
# This script requires a modified version of [`keras_transformer`](https://github.com/kpot/keras-transformer), which was developed by [Kirill Mavreshko](https://github.com/kpot). The modified version can be installed from source as follows:
# 
# ```
# git clone https://github.com/klarh/keras-transformer
# pip install -e ./keras-transformer
# ```
# 
# This script utilizes lightly modified code from the `keras_transformer` examples to generate the wikitext-2 data and build the language model. The `keras_transformer` project is made available under the MIT license, reproduced below.
# 
# <div style="font-size: 80%; font-family: monospace;"><pre>
# The MIT License
# 
# Copyright 2018 Kirill Mavreshko (https://www.linkedin.com/in/kirill-mavreshko/)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# </pre></div>
#This script is a modified version of SI_Notebook_B._Language_Modeling.ipynb

import os
#tf logging level, see https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information/36614636
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import logging
import numpy as np
import keras
import keras.backend as K
import matplotlib, matplotlib.pyplot as pp
import sys
from time import time
import tensorflow as tf
import timeit
import collections
import itertools
import tqdm
import keras_tqdm
import pandas as pd

import keras_transformer
import keras_transformer.attention
from keras.layers import Input, Softmax, Embedding, Add, Lambda, Dense
from keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from keras_transformer.position import TransformerCoordinateEmbedding
from keras_transformer.transformer import TransformerACT, TransformerBlock
from keras.utils import multi_gpu_model

from itertools import islice
from typing import Iterable, List, Optional

from example import wikitext
from example.bpe import BPEEncoder, ID_FOR_PADDING


tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)



tqdm.tqdm = tqdm.tqdm_notebook

# assume that the keras_transformer package is installed
# in developer mode (used when we import example later)
sys.path.append(os.path.join(os.path.dirname(keras_transformer.__file__), '..'))

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
#tf_config.gpu_options.per_process_gpu_memory_fraction=0.9
tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
session = tf.Session(config=tf_config)
K.set_session(session)

def perplexity(y_true, y_pred):
    """
    Popular metric for evaluating language modelling architectures.
    More info: http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf
    """
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    return K.mean(K.exp(K.mean(cross_entropy, axis=-1)))

def bpc(y_true, y_pred):
    """Bits per character metric, commonly used for compression-type tasks."""
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    return K.mean(cross_entropy)/np.log(2)

class TimingCallback(keras.callbacks.Callback):
    """Measure the time required for each epoch"""
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time()
        
    def on_epoch_end(self, epoch, logs={}):
        logs['epoch_time'] = time() - self.starttime
        
def universal_transformer_gpt_model(
        max_seq_length: int, vocabulary_size: int,
        word_embedding_size: int, transformer_depth: int,
        num_heads: int, transformer_dropout: float = 0.1,
        embedding_dropout: float = 0.6,
        l2_reg_penalty: float = 1e-6,
        confidence_penalty_weight: float = 0.1,
        agglomerative_attention: bool = False,
        use_convolutions: bool = False,
        use_coordinate_embeddings: bool = True,
        convolution_width: int = 0,
        penalize_confidence: bool = False):
    """
    A model which is similar to the one described by OpenAI in paper
    "Improving Language Understanding by Generative Pre-Training", except
    that it relies L2 regularization of the word embedding matrix
    (instead of the dropout), and uses Universal Transformer architecture.
    Derived from the keras-transformer project examples.
    """
    word_ids = Input(shape=(max_seq_length,), dtype='int32', name='word_ids')
    l2_regularizer = (keras.regularizers.l2(l2_reg_penalty) if l2_reg_penalty
                      else None)
    embedding_layer = ReusableEmbedding(
        vocabulary_size, word_embedding_size,
        input_length=max_seq_length,
        name='bpe_embeddings',
        # Regularization is based on paper "A Comparative Study on
        # Regularization Strategies for Embedding-based Neural Networks"
        # https://arxiv.org/pdf/1508.03721.pdf
        embeddings_regularizer=l2_regularizer)
    output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='word_prediction_logits')
    conv_layer = keras.layers.Conv1D(
        word_embedding_size, convolution_width, padding='causal',
        activation='relu', kernel_initializer='he_uniform', name='convolution')
    coordinate_embedding_layer = TransformerCoordinateEmbedding(
        transformer_depth,
        name='coordinate_embedding')
    transformer_block = TransformerBlock(
        name='transformer', num_heads=num_heads,
        residual_dropout=transformer_dropout,
        attention_dropout=transformer_dropout,
        use_masking=True, vanilla_wiring=False,
        agglomerative_attention=agglomerative_attention)
    transformer_act_layer = TransformerACT(name='adaptive_computation_time')
    output_softmax_layer = Softmax(name='word_predictions')

    next_step_input, embedding_matrix = embedding_layer(word_ids)
    act_output = next_step_input

    for i in range(transformer_depth):
        if use_convolutions:
            next_step_input = conv_layer(next_step_input)
        if use_coordinate_embeddings:
            next_step_input = coordinate_embedding_layer(next_step_input, step=i)
        next_step_input = transformer_block(next_step_input)
        next_step_input, act_output = transformer_act_layer(next_step_input)

    transformer_act_layer.finalize()
    next_step_input = act_output
    word_predictions = output_softmax_layer(
        output_layer([next_step_input, embedding_matrix]))
    model = keras.models.Model(inputs=[word_ids], outputs=[word_predictions])
    # Penalty for confidence of the output distribution, as described in
    # "Regularizing Neural Networks by Penalizing Confident
    # Output Distributions" (https://arxiv.org/abs/1701.06548)
    confidence_penalty = K.mean(
        confidence_penalty_weight *
        K.sum(word_predictions * K.log(word_predictions), axis=-1))
    if penalize_confidence:
        model.add_loss(confidence_penalty)
    return model        
        
def get_model(max_seq_length, agglomerative_attention, 
              vocabulary_size, embedding_size, num_heads, 
              use_convolutions=False, 
              use_coordinate_embeddings=True,
              convolution_width=0,
              adjust_lr=False, num_gpus=1):
    
    optimizer = keras.optimizers.adadelta()
    model = universal_transformer_gpt_model(
        max_seq_length=max_seq_length,
        vocabulary_size=vocabulary_size,
        word_embedding_size=embedding_size,
        transformer_depth=5,
        num_heads=num_heads,
        agglomerative_attention=agglomerative_attention,
        use_convolutions=use_convolutions,
        use_coordinate_embeddings=use_coordinate_embeddings,
        convolution_width=convolution_width)
    if num_gpus>1:
        model = multi_gpu_model(model, gpus=num_gpus)

    model.compile(
        optimizer,
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=[perplexity, bpc])
        
    model_callbacks = [
        TimingCallback(),
        keras_tqdm.TQDMNotebookCallback(show_inner=False),
        keras.callbacks.EarlyStopping(
            'val_loss', patience=10, restore_best_weights=True),
    ]
    
    if adjust_lr:
        model_callbacks.append(
            keras.callbacks.ReduceLROnPlateau(factor=.9, patience=3))
    
    return model, model_callbacks


# # WikiText-2 Dataset
def pad_lm_samples(samples: Iterable[List[int]],
                   required_sequence_length: int):
    tail_padding = [ID_FOR_PADDING]
    for sample in samples:
        assert len(sample) > 0
        sample.extend(tail_padding * (required_sequence_length - len(sample)))

def training_data_to_samples(training_set_name: str,
                             encoder: BPEEncoder,
                             max_sequence_length: int) -> np.ndarray:
    """
    Reads WikiText dataset, interpreting each line as an independent sequence,
    then splits those lines with BPE tokenizer and turns them into word ids
    based on previously constructed BPE vocabulary (both the tokenizer
    and the vocabulary are parts of the BPEEncoder instance).

    Those word id's then packed into a matrix the size of
    (number of lines x max_sequence_length + 1), which can be later sliced
    to get X and Y matrices of sequences for training).
    """
    training_set = wikitext.read_wikitext_file(training_set_name)
    useful_sequences = []
    for line in training_set.splitlines():
        clean_line = line.strip()
        is_header = clean_line.startswith('=') and clean_line.endswith('=')
        if is_header or not clean_line:
            continue
        # the encoder is supposed to add <SEQ> and </SEQ>
        id_word_pairs = list(encoder(clean_line))
        useful_sequences.append(
            [word_id for word_id, _ in id_word_pairs[:max_sequence_length]])

    pad_lm_samples(useful_sequences, max_sequence_length + 1)
    result = np.empty(
        (len(useful_sequences), max_sequence_length + 1),
        dtype='int32')
    for i, sequence in enumerate(useful_sequences):
        result[i, :] = sequence
    return result

def training_data_to_dense_samples(training_set_name: str,
                                   encoder: BPEEncoder,
                                   max_sequence_length: int) -> np.ndarray:
    """
    Reads WikiText dataset, interpreting each line as an independent sequence,
    then splits those lines with BPE tokenizer and turns them into word ids
    based on previously constructed BPE vocabulary (both the tokenizer
    and the vocabulary are parts of the BPEEncoder instance).

    Those word id's then packed into a matrix the size of
    (number of lines x max_sequence_length + 1), which can be later sliced
    to get X and Y matrices of sequences for training).
    """
    training_set = wikitext.read_wikitext_file(training_set_name)
    useful_sequences = []

    def stream_bpe_tokens():
        for line in training_set.splitlines():
            clean_line = line.strip()
            if not clean_line:
                continue
            # the encoder is supposed to add <SEQ> and </SEQ>
            id_word_pairs = encoder(clean_line)
            yield from id_word_pairs

    id_word_stream = stream_bpe_tokens()
    while True:
        chunk = list(islice(id_word_stream, max_sequence_length))
        if len(chunk) == 0:
            break
        sample_sequence = [word_id for word_id, _ in chunk]
        useful_sequences.append(sample_sequence)

    pad_lm_samples(useful_sequences, max_sequence_length + 1)
    result = np.empty(
        (len(useful_sequences), max_sequence_length + 1),
        dtype='int32')
    for i, sequence in enumerate(useful_sequences):
        result[i, :] = sequence
    return result

def train_wikitext2(args, agglomerative_attention,
                    use_convolutions=False, use_coordinate_embeddings=True):
    print('>>train_wikitext2()')
    encoder = wikitext.build_wikitext_bpe_encoder()

    def x_y_for_dataset(dataset_name):
        fat_sample = training_data_to_dense_samples(
            dataset_name, encoder, args.wt2_seq_len)
        _x = fat_sample[:, :args.wt2_seq_len]
        _y = np.expand_dims(fat_sample[:, 1:], axis=-1)
        return _x, _y

    x, y = x_y_for_dataset(wikitext.TRAINING_SET_NAME)

    model, model_callbacks = get_model(args.wt2_seq_len, agglomerative_attention,
          encoder.vocabulary_size(), args.wt2_embedding_size, args.wt2_num_heads,
          use_convolutions, use_coordinate_embeddings, convolution_width=8, num_gpus=args.num_gpus)
    print(model.summary())

    model.fit(
        x, y,
        validation_data=x_y_for_dataset(wikitext.VALIDATION_SET_NAME),
        batch_size=args.wt2_batch_size, epochs=args.wt2_epochs, verbose=args.train_verbosity,
        callbacks=model_callbacks)

    model_save_path = f"models/wiki2_{args.wt2_batch_size}_{args.wt2_seq_len}_{args.wt2_embedding_size}_{args.wt2_num_heads}_{args.wt2_epochs}.h5"
    # Save model via the template model (which shares the same weights):
    model.save(model_save_path)

    test_evals = model.evaluate(
        *x_y_for_dataset(wikitext.TEST_SET_NAME),
        verbose=False, batch_size=args.wt2_batch_size)
    
    return model, test_evals


def run_train_wiki2(args):
    print('>>run_train_wiki2()')
    aggloms = [False, True]
    convolutions = [False, True]
    coords = [False, True]
    replicas = list(range(args.wt2_replicas))

    wikitext2_histories = collections.defaultdict(list)
    wikitext2_sizes = {}
    wikitext2_test_evals = collections.defaultdict(list)

    for (agglom, conv, coord, _) in itertools.product(
            aggloms, convolutions, coords, replicas):
        # skip models that won't include any notion of time
        if not coord and not conv:
            continue
        # also skip models that include both
        if coord and conv:
            continue

        key = (agglom, conv, coord)

        (model, test_evals) = train_wikitext2(args, agglom, use_convolutions=conv, use_coordinate_embeddings=coord)

        wikitext2_sizes[key] = sum(v.size for v in model.get_weights())
        wikitext2_histories[key].append(model.history.history)
        wikitext2_test_evals[key].append(test_evals)
    return wikitext2_histories, wikitext2_test_evals, wikitext2_sizes,model,key

def plot_wiki2_loss(wikitext2_histories, last_key):
    names = list(wikitext2_histories[last_key][0])

    for name in names:
        for key in list(sorted(wikitext2_histories)):
            (agglom, conv, coord) = key
            label = 'agg' if agglom else 'full'
            if conv:
                label += ', conv'
            if coord:
                label += ', coord'

            histories = [hist[name] for hist in wikitext2_histories[key]]
            N = min(len(hist) for hist in histories)
            histories = np.array([hist[:N] for hist in histories])
            mu = np.mean(histories, axis=0)
            pp.plot(mu, label=label)
        pp.title(name)
        pp.legend()
        pp.show(block=False)
        pp.savefig('media/wiki2_loss.pdf')

def show_wiki2_df(wikitext2_histories, wikitext2_test_evals, wikitext2_sizes, model):
    rows = []

    for key in list(sorted(wikitext2_test_evals)):
        (agglom, conv, coord) = key
        label = 'agg' if agglom else 'full'
        if conv:
            label += ', conv'
        if coord:
            label += ', coord'

        epoch_times = [hist['epoch_time'] for hist in wikitext2_histories[key]]
        epoch_time = np.mean([np.mean(t[-len(t)//2:]) for t in epoch_times])

        row = [label, wikitext2_sizes[key], epoch_time]

        row.extend(np.mean(wikitext2_test_evals[key], axis=0).tolist())
        row.extend((np.std(wikitext2_test_evals[key], axis=0)/
                    np.sqrt(len(wikitext2_test_evals[key]))).tolist())

        rows.append(row)

    columns = (['Name', 'size', 'epoch time'] +
               model.metrics_names +
               [name + '_stderr' for name in model.metrics_names])
    df = pd.DataFrame(rows, columns=columns)
    print(df)
    df.to_csv('media/wiki2_train.csv', index=False)

def plot_wiki2_training(wikitext2_histories):
    # add to main
    colors = pp.rcParams['axes.prop_cycle'].by_key()['color']

    pp.figure(figsize=(2, 1)*np.array(pp.rcParams['figure.figsize']))
    ax = pp.subplot(1, 2, 1);name = 'loss'
    for (key, color) in zip(list(sorted(wikitext2_histories)), colors[2:]):
        (agglom, conv, coord) = key
        label = 'agg.' if agglom else 'full'
        if conv:
            label += ', conv.'
        if coord:
            label += ', coord.'

        histories = [hist[name] for hist in wikitext2_histories[key]]
        N = min(len(hist) for hist in histories)
        histories = np.array([hist[:N] for hist in histories])
        mu = np.mean(histories, axis=0)
        ax.plot(mu, label=label, color=color)

        val_name = 'val_{}'.format(name)
        histories = [hist[val_name] for hist in wikitext2_histories[key]]
        N = min(len(hist) for hist in histories)
        histories = np.array([hist[:N] for hist in histories])
        mu = np.mean(histories, axis=0)
        ax.plot(mu, linestyle='--', color=color)
    pp.ylabel(name.capitalize())
    pp.xlabel('Epoch')
    # pp.gca().set_xscale('log')
    # pp.gca().set_yscale('log')
    pp.legend()

    ax = pp.subplot(1, 2, 2);name = 'perplexity'
    for (key, color) in zip(list(sorted(wikitext2_histories)), colors[2:]):
        (agglom, conv, coord) = key
        label = 'agg.' if agglom else 'full'
        if conv:
            label += ', conv.'
        if coord:
            label += ', coord.'

        histories = [hist[name] for hist in wikitext2_histories[key]]
        N = min(len(hist) for hist in histories)
        histories = np.array([hist[:N] for hist in histories])
        mu = np.mean(histories, axis=0)
        ax.plot(mu, label=label, color=color)

        val_name = 'val_{}'.format(name)
        histories = [hist[val_name] for hist in wikitext2_histories[key]]
        N = min(len(hist) for hist in histories)
        histories = np.array([hist[:N] for hist in histories])
        mu = np.mean(histories, axis=0)
        ax.plot(mu, linestyle='--', color=color)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    pp.ylabel(name.capitalize())
    pp.xlabel('Epoch')
    # pp.gca().set_xscale('log')
    # pp.gca().set_yscale('log')
    pp.legend()
    pp.savefig('media/wikitext2_training.pdf')

def random_batch(data, compress_character_map, batch_size, seq_len, use_fractions=(0, 1.)):
    N = len(data)
    whole_start_index = int(N*use_fractions[0])
    whole_end_index = int(N*use_fractions[1])
    
    while True:
        start_indices = np.random.randint(
            whole_start_index, whole_end_index - seq_len, 
            size=batch_size)
        end_indices = start_indices + seq_len + 1

        slices = np.array(
            [compress_character_map[data[start:end]]
             for (start, end) in zip(start_indices, end_indices)],
            dtype=data.dtype)

        inputs = slices[:, :-1]
        outputs = slices[:, 1:, np.newaxis]
        yield (inputs, outputs)
        
def train_text8(vocabulary_size, text8_data, agglomerative_attention,
                compress_character_map, use_convolutions=True,
                use_coordinate_embeddings=True):
    '''Character level language model training'''
    print('>>train_text8()')

    model, model_callbacks = get_model(
        args.text8_seq_len, agglomerative_attention,
        vocabulary_size, args.text8_embedding_size, args.text8_num_heads,
        use_convolutions=use_convolutions, 
        use_coordinate_embeddings=use_coordinate_embeddings,
        convolution_width=8)
    print(model.summary())

    fractions = 1 - np.cumsum([args.text8_test_frac, args.text8_val_frac])[::-1]
    train_data = random_batch(text8_data, compress_character_map, args.text8_batch_size, args.text8_seq_len, (0, fractions[0]))
    val_data = random_batch(text8_data, compress_character_map, args.text8_batch_size, args.text8_seq_len, (fractions[0], fractions[1]))
    test_data = random_batch(text8_data, compress_character_map, args.text8_batch_size, args.text8_seq_len, (fractions[1], 1))
    
    steps_per_epoch = int(args.text8_epoch_scaling_factor*fractions[0]*
                          len(text8_data)/args.text8_seq_len/args.text8_batch_size)
    validation_steps = int(steps_per_epoch*
                           args.text8_val_frac/fractions[0])
    test_steps = int(steps_per_epoch*
                     args.text8_test_frac/fractions[0])

    model.fit_generator(
        train_data, 
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        validation_steps=validation_steps,
        epochs=args.text8_epochs,
        callbacks=model_callbacks,
        verbose=args.train_verbosity,
    )

    model_save_path = f"models/text8_{args.text8_batch_size}_{args.text8_seq_len}_{args.text8_embedding_size}_{args.text8_num_heads}_{args.text8_epochs}.h5"
    model.save(model_save_path)
    
    test_evals = model.evaluate_generator(
        test_data, 
        steps=test_steps,
        verbose=False)
    
    return model, test_evals

def text8_proc_data(args):
    print('>>text8_proc_data()')
    # # Text8 dataset

    #run these separately
    #get_ipython().run_cell_magic('writefile', 'text8_conversion.pl', '#!/usr/bin/perl\n\n# Program to filter Wikipedia XML dumps to "clean" text consisting only of lowercase\n# letters (a-z, converted from A-Z), and spaces (never consecutive).  \n# All other characters are converted to spaces.  Only text which normally appears \n# in the web browser is displayed.  Tables are removed.  Image captions are \n# preserved.  Links are converted to normal text.  Digits are spelled out.\n\n# Written by Matt Mahoney, June 10, 2006.  This program is released to the public domain.\n\n$/=">";                     # input record separator\nwhile (<>) {\n  if (/<text /) {$text=1;}  # remove all but between <text> ... </text>\n  if (/#redirect/i) {$text=0;}  # remove #REDIRECT\n  if ($text) {\n\n    # Remove any text not normally visible\n    if (/<\\/text>/) {$text=0;}\n    s/<.*>//;               # remove xml tags\n    s/&amp;/&/g;            # decode URL encoded chars\n    s/&lt;/</g;\n    s/&gt;/>/g;\n    s/<ref[^<]*<\\/ref>//g;  # remove references <ref...> ... </ref>\n    s/<[^>]*>//g;           # remove xhtml tags\n    s/\\[http:[^] ]*/[/g;    # remove normal url, preserve visible text\n    s/\\|thumb//ig;          # remove images links, preserve caption\n    s/\\|left//ig;\n    s/\\|right//ig;\n    s/\\|\\d+px//ig;\n    s/\\[\\[image:[^\\[\\]]*\\|//ig;\n    s/\\[\\[category:([^|\\]]*)[^]]*\\]\\]/[[$1]]/ig;  # show categories without markup\n    s/\\[\\[[a-z\\-]*:[^\\]]*\\]\\]//g;  # remove links to other languages\n    s/\\[\\[[^\\|\\]]*\\|/[[/g;  # remove wiki url, preserve visible text\n    s/\\{\\{[^}]*\\}\\}//g;         # remove {{icons}} and {tables}\n    s/\\{[^}]*\\}//g;\n    s/\\[//g;                # remove [ and ]\n    s/\\]//g;\n    s/&[^;]*;/ /g;          # remove URL encoded chars\n\n    # convert to lowercase letters and spaces, spell digits\n    $_=" $_ ";\n    tr/A-Z/a-z/;\n    s/0/ zero /g;\n    s/1/ one /g;\n    s/2/ two /g;\n    s/3/ three /g;\n    s/4/ four /g;\n    s/5/ five /g;\n    s/6/ six /g;\n    s/7/ seven /g;\n    s/8/ eight /g;\n    s/9/ nine /g;\n    tr/a-z/ /cs;\n    chop;\n    print $_;\n  }\n}')
    #get_ipython().run_cell_magic('sh', '', '\n# see http://mattmahoney.net/dc/textdata.html\nwget --continue http://mattmahoney.net/dc/enwik9.zip\n\nunzip -p enwik9.zip enwik9 | perl text8_conversion.pl | head -c 100000000 > text8\n\nls -lh')

    text8_data = np.memmap(args.text8_data, mode='r')

    # remove spaces in the ASCII character set, given that we have a-z and ' '
    # reserve index 0 for masking later, if desired
    compressed_characters = [0]

    compressed_characters.extend(range(ord('a'), ord('z') + 1))
    compressed_characters.append(ord(' '))

    compressed_characters = np.array(compressed_characters, dtype=np.uint8)

    compress_character_map = np.zeros(256, dtype=np.uint8)
    for (i, j) in enumerate(compressed_characters):
        compress_character_map[j] = i

    vocabulary_size = len(compressed_characters)
    return text8_data, vocabulary_size, compress_character_map

def run_train_text8(args, text8_data, vocabulary_size, compress_character_map):
    print('>>run_train_text8()')
    aggloms = [False, True]
    convolutions = [False, True]
    coords = [False, True]
    replicas = list(range(args.text8_replicas))

    text8_histories = collections.defaultdict(list)
    text8_sizes = {}
    text8_test_evals = collections.defaultdict(list)

    for (agglom, conv, coord, _) in itertools.product(
            aggloms, convolutions, coords, replicas):
        # skip models that won't include any notion of time
        if not coord and not conv:
            continue
        # also skip models that include both
        if coord and conv:
            continue

        key = (agglom, conv, coord)

        (model, test_evals) = train_text8(vocabulary_size=vocabulary_size,
                                          text8_data=text8_data,
                                          agglomerative_attention=agglom,
                                          compress_character_map=compress_character_map,
                                          use_convolutions=conv,
                                          use_coordinate_embeddings=coord)

        text8_histories[key].append(model.history.history)
        text8_sizes[key] = sum(v.size for v in model.get_weights())
        text8_test_evals[key].append(test_evals)
    return text8_histories, text8_sizes, text8_test_evals, model, key

def plot_txt8_loss(text8_histories, key):
    names = list(text8_histories[key][0])

    for name in names:
        for key in list(sorted(text8_histories)):
            (agglom, conv, coord) = key
            label = 'agg' if agglom else 'full'
            if conv:
                label += ', conv'
            if coord:
                label += ', coord'

            histories = [hist[name] for hist in text8_histories[key]]
            N = min(len(hist) for hist in histories)
            histories = np.array([hist[:N] for hist in histories])
            mu = np.mean(histories, axis=0)
            pp.plot(mu, label=label)
        pp.title(name)
        pp.legend()
        pp.show(block=False)
        pp.savefig('media/text8_loss.png')

def show_text8_df(text8_histories, text8_sizes, text8_test_evals, model):
    rows = []

    for key in list(sorted(text8_test_evals)):
        (agglom, conv, coord) = key
        label = 'agg' if agglom else 'full'
        if conv:
            label += ', conv'
        if coord:
            label += ', coord'

        epoch_times = [hist['epoch_time'] for hist in text8_histories[key]]
        epoch_time = np.mean([np.mean(t[-len(t)//2:]) for t in epoch_times])

        row = [label, text8_sizes[key], epoch_time]

        row.extend(np.mean(text8_test_evals[key], axis=0).tolist())
        row.extend((np.std(text8_test_evals[key], axis=0)/
                    np.sqrt(len(text8_test_evals[key]))).tolist())

        rows.append(row)

    columns = (['Name', 'size', 'epoch time'] +
               model.metrics_names +
               [name + '_stderr' for name in model.metrics_names])
    df=pd.DataFrame(rows, columns=columns)
    print(df)
    df.to_csv('text8_train.csv', index=False)

def plot_text8_training(text8_histories):

    colors = pp.rcParams['axes.prop_cycle'].by_key()['color']

    name = 'bpc'
    for (key, color) in zip(list(sorted(text8_histories)), colors[2:]):
        (agglom, conv, coord) = key
        label = 'agg.' if agglom else 'full'
        if conv:
            label += ', conv.'
        if coord:
            label += ', coord.'

        histories = [hist[name] for hist in text8_histories[key]]
        N = min(len(hist) for hist in histories)
        histories = np.array([hist[:N] for hist in histories])
        mu = np.mean(histories, axis=0)
        pp.plot(mu, label=label, color=color)

        val_name = 'val_{}'.format(name)
        histories = [hist[val_name] for hist in text8_histories[key]]
        N = min(len(hist) for hist in histories)
        histories = np.array([hist[:N] for hist in histories])
        mu = np.mean(histories, axis=0)
        pp.plot(mu, linestyle='--', color=color)

    pp.ylabel('Loss / BPC')
    pp.xlabel('Epoch')
    # pp.gca().set_xscale('log')
    # pp.gca().set_yscale('log')
    pp.legend()
    pp.savefig('media/text8_training.pdf')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train_wt2", action='store_true', help="Whether to run wikitext2 training.")
    parser.add_argument("--do_train_text8", action='store_true', help="Whether to run text8 training.")
    parser.add_argument('--wt2_batch_size', type=int, default=32)
    parser.add_argument('--wt2_seq_len', type=int, default=128)
    parser.add_argument('--wt2_epochs', type=int, default=400)
    parser.add_argument('--wt2_embedding_size', type=int, default=128)
    parser.add_argument('--wt2_num_heads', type=int, default=8)
    parser.add_argument('--wt2_replicas', type=int, default=1)

    parser.add_argument('--text8_embedding_size', type=int, default=64)
    parser.add_argument('--text8_num_heads', type=int, default=8)
    parser.add_argument('--text8_data', type=str, default='data/text8')
    parser.add_argument('--text8_batch_size', type=int, default=64)
    parser.add_argument('--text8_seq_len', type=int, default=128)
    parser.add_argument('--text8_epochs', type=int, default=400)
    parser.add_argument('--text8_test_frac', type=float, default=0.1)
    parser.add_argument('--text8_val_frac', type=float, default=0.3)
    parser.add_argument('--text8_epoch_scaling_factor', type=int, default=1)
    parser.add_argument('--text8_replicas', type=int, default=1)

    parser.add_argument('--train_verbosity', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()
    print(args)
    return args


def wiki2_workflow(args):
    print('training on Wiki2 dataset')
    wikitext2_histories, wikitext2_test_evals, wikitext2_sizes,model,key=run_train_wiki2(args)
    plot_wiki2_loss(wikitext2_histories, last_key=key)
    show_wiki2_df(wikitext2_histories, wikitext2_test_evals, wikitext2_sizes, model)
    plot_wiki2_training(wikitext2_histories)


def text8_workflow(args):
    print('training on Text8 dataset')
    text8_data, vocabulary_size, compress_character_map=text8_proc_data(args)
    text8_histories, text8_sizes, text8_test_evals, model, key=run_train_text8(args, text8_data=text8_data, vocabulary_size=vocabulary_size, compress_character_map=compress_character_map)
    plot_txt8_loss(text8_histories, last_key=key)
    show_text8_df(text8_histories, text8_sizes, text8_test_evals, model)
    plot_text8_training(text8_histories)

if __name__ == '__main__':
    args = get_args()
    if args.do_train_wt2:
        wiki2_workflow(args)
    if args.do_train_text8:
        text8_workflow(args)