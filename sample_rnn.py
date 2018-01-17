from __future__ import print_function

import os
import sys
import time
import importlib
if sys.version_info < (3,0):
    import cPickle as pickle
else:
    import pickle

from folk_rnn import Folk_RNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('metadata_path')
parser.add_argument('--rng_seed', type=int)
parser.add_argument('--temperature', type=float)
parser.add_argument('--ntunes', type=int, default=1)
parser.add_argument('--seed')
parser.add_argument('--terminal', action="store_true")

args = parser.parse_args()

metadata_path = args.metadata_path
rng_seed = args.rng_seed
temperature = args.temperature
ntunes = args.ntunes
seed = args.seed

print('seed', seed)

with open(metadata_path, 'rb') as f:
    if sys.version_info < (3,0): 
        metadata = pickle.load(f)
    else:
        metadata = pickle.load(f, encoding='latin1')

config = importlib.import_module('configurations.%s' % metadata['configuration'])

# samples dir
if not os.path.isdir('samples'):
        os.makedirs('samples')
target_path = "samples/%s-s%d-%.2f-%s.txt" % (
    metadata['experiment_id'], rng_seed, temperature, time.strftime("%Y%m%d-%H%M%S", time.localtime()))

folk_rnn = Folk_RNN(
    metadata['token2idx'],
    metadata['param_values'], 
    config.num_layers, 
    )
folk_rnn.seed_tune(seed)
for i in range(ntunes):
    tune_tokens = folk_rnn.generate_tune(random_number_generator_seed=rng_seed, temperature=temperature)
    tune = 'X:{}\n{}\n{}\n{}\n'.format(i, tune_tokens[0], tune_tokens[1], ' '.join(tune_tokens[2:]))
    if args.terminal:
        print(tune)
    else:
        with open(target_path, 'a+') as f:
            f.write(tune)
        print('Saved to {}'.format(target_path))
    
    
