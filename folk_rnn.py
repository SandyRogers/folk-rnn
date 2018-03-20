import numpy as np
from inspect import signature

TUNE_MAX_LENGTH = 1000

def sigmoid(x): 
    return 1/(1 + np.exp(-x))

def softmax(x,T): 
    expx=np.exp(x/T)
    sumexpx=np.sum(expx)
    if sumexpx==0:
       maxpos=x.argmax()
       x=np.zeros(x.shape, dtype=x.dtype)
       x[0][maxpos]=1
    else:
       x=expx/sumexpx
    return x

class Folk_RNN:
    """
    Folk music style modelling using LSTMs
    """
    
    def __init__(self, token2idx, param_values, num_layers=3, wildcard_token=None):
        vocab_size = len(token2idx)
        self.num_layers = num_layers
        self.token2idx = token2idx
        if wildcard_token is not None:
            self.token2idx[wildcard_token] = vocab_size
        self.idx2token = dict((v, k) for k, v in self.token2idx.items())
        self.vocab_idxs = np.arange(vocab_size)
        self.start_idx, self.end_idx = self.token2idx['<s>'], self.token2idx['</s>']
        self.wildcard_idx = self.token2idx[wildcard_token]
        
        layer_indexes = range(self.num_layers)
        self.LSTM_Wxi = [param_values[2+jj*14-1] for jj in layer_indexes]
        self.LSTM_Whi = [param_values[3+jj*14-1] for jj in layer_indexes]
        self.LSTM_bi =  [param_values[4+jj*14-1] for jj in layer_indexes]
        self.LSTM_Wxf =[param_values[5+jj*14-1] for jj in layer_indexes]
        self.LSTM_Whf = [param_values[6+jj*14-1] for jj in layer_indexes]
        self.LSTM_bf = [param_values[7+jj*14-1] for jj in layer_indexes]
        self.LSTM_Wxc = [param_values[8+jj*14-1] for jj in layer_indexes]
        self.LSTM_Whc = [param_values[9+jj*14-1] for jj in layer_indexes]
        self.LSTM_bc = [param_values[10+jj*14-1] for jj in layer_indexes]
        self.LSTM_Wxo = [param_values[11+jj*14-1] for jj in layer_indexes]
        self.LSTM_Who = [param_values[12+jj*14-1] for jj in layer_indexes]
        self.LSTM_bo = [param_values[13+jj*14-1] for jj in layer_indexes]
        self.LSTM_cell_init = [param_values[14+jj*14-1] for jj in layer_indexes]
        self.LSTM_hid_init = [param_values[15+jj*14-1] for jj in layer_indexes]
        
        self.FC_output_W = param_values[43];
        self.FC_output_b = param_values[44];
        
        self.sizeofx=self.LSTM_Wxi[0].shape[0]
        
        self.seed_tune()
    
    def seed_tune(self, seed_tune_abc=None):
        """
        Sets the seed of the tune. Accepts either an iterable of tokens, or string of tokens separated by tokens.
        """
        self.tune = [self.start_idx]
        if isinstance(seed_tune_abc, str):
            self.tune += [self.token2idx[x] for x in seed_tune_abc.split(' ')]
        elif seed_tune_abc:
            self.tune += [self.token2idx[x] for x in seed_tune_abc]
    
    def generate_tune(self, random_number_generator_seed=42, temperature=1.0, on_token_callback=None):
        """
        Generates tune and returns it as a list of abc tokens
        """
        callback_type = len(signature(on_token_callback).parameters) if on_token_callback else 0
        tune = list(self.tune)
        htm1 = list(self.LSTM_hid_init)
        ctm1 = list(self.LSTM_cell_init)
        rng = np.random.RandomState(random_number_generator_seed)
        token_count = 0
        seed_count = len(self.tune)
        while tune[-1] != self.end_idx and token_count < TUNE_MAX_LENGTH:
            x = np.zeros(self.sizeofx, dtype=np.int8)
            x[tune[token_count]] = 1;
            for jj in range(self.num_layers):
               it=sigmoid(np.dot(x,self.LSTM_Wxi[jj]) + np.dot(htm1[jj], self.LSTM_Whi[jj]) + self.LSTM_bi[jj])
               ft=sigmoid(np.dot(x,self.LSTM_Wxf[jj]) + np.dot(htm1[jj], self.LSTM_Whf[jj]) + self.LSTM_bf[jj])
               ct=np.multiply(ft,ctm1[jj]) + np.multiply(it,np.tanh(np.dot(x,self.LSTM_Wxc[jj]) + np.dot(htm1[jj],self.LSTM_Whc[jj]) + self.LSTM_bc[jj]))
               ot=sigmoid(np.dot(x,self.LSTM_Wxo[jj]) + np.dot(htm1[jj],self.LSTM_Who[jj]) + self.LSTM_bo[jj])
               ht=np.multiply(ot,np.tanh(ct))
               x=ht
               ctm1[jj]=ct
               htm1[jj]=ht
            output = softmax(np.dot(x,self.FC_output_W) + self.FC_output_b, temperature).squeeze()

            next_itoken = rng.choice(self.vocab_idxs, p=output)
            next_itoken_alternatives = sorted(zip(self.vocab_idxs, output), key=lambda x: x[1], reverse=True)
            
            token_count += 1
            if token_count >= seed_count:
                tune.append(next_itoken)
            elif tune[token_count] == self.wildcard_idx:
                tune[token_count] = next_itoken
            
            if callback_type == 1 and next_itoken != self.end_idx:
                on_token_callback(self.idx2token[tune[token_count]])    
            elif callback_type == 2 and next_itoken != self.end_idx:
                on_token_callback(self.idx2token[tune[token_count]], ((self.idx2token[i], p) for i,p in next_itoken_alternatives))
            
        return [self.idx2token[x] for x in tune[1:-1]]
