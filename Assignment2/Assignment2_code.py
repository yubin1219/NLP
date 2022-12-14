import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time

# Set random seeds for reproducability
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

"""
!python -m spacy download en_core_web_sm
!python -m spacy download de_core_news_sm
"""

# Load the German spaCy model
spacy_de = spacy.load('de_core_news_sm')
# Load the English spaCy model
spacy_en = spacy.load('en_core_web_sm')

# Create Tokenizers
"""
Tokenizer is used to turn a string containing a sentence into a list of individual tokens that make up that string.
e.g. "I have a dream!" becomes ["I", "have", "a", "dream", "!"]
Take in the sentence as a string and return the sentence as a list of tokens.
"""
def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

"""
Set the tokenize argument to the correct tokenization function for each.
The model expects data to be fed in with the batch dimension first, so we use batch_first = True, and convert all words to lowercase.
"""
# Source field
SRC = Field(tokenize = tokenize_de, 
            init_token = '', 
            eos_token = '', 
            lower = True,
            batch_first=True)
# Target field
TRG = Field(tokenize = tokenize_en, 
            init_token = '', 
            eos_token = '', 
            lower = True,
            batch_first=True)

# Load datasets and split
train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
                                                    fields = (SRC, TRG))

# Build the vocabulary for the source and target languages
"""
The vocabularies of the source and target languages are distinct.
Using the min_freq argument, we only allow tokens that appear at least 2 times to appear in our vocabulary.
Tokens that appear only once are converted into an <unk> (unknown) token.
"""
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

# Create the iterators
"""
All of the source sentences are padded to the same length, the same with the target sentences.
Use a BucketIterator instead of the standard Iterator as it creates batches in such a way that it minimizes the amount of padding in both the source and target sentences.
"""
BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        # src : [batch size, src len]
        # src_mask : [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device) # [batch size, src len]
        
        # positional embedding??? embedding??? ??? token??? summation?????? ?????? ??? ???????????? ?????? ????????? ??????
        # ?????? ???????????? ?????? ??? ????????? ?????? ???????????? ????????? ?????? ??? ??????        
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)) # [batch size, src len, hid dim]
               
        for layer in self.layers:
            src = layer(src, src_mask) # [batch size, src len, hid dim]
                        
        return src

class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        # src : [batch size, src len, hid dim]
        # src_mask : [batch size, 1, 1, src len] 
                
        # Multi-Head Self Attention ??????
        # ????????? sentence ?????? ?????? ?????? token?????? ???????????? ??? token?????? representation??? ??????
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src)) # [batch size, src len, hid dim]
      
        # positionwise feedforward - ?????? ????????? features??? refine?????? ??????
        _src = self.positionwise_feedforward(src)
        
        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src)) # [batch size, src len, hid dim]
                
        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        # query : [batch size, query len, hid dim]
        # key : [batch size, key len, hid dim]
        # value : [batch size, value len, hid dim]
        batch_size = query.shape[0]
        
        # query vector, key vector, value vector ??????
        Q = self.fc_q(query) # [batch size, query len, hid dim]
        K = self.fc_k(key)   # [batch size, key len, hid dim]
        V = self.fc_v(value) # [batch size, value len, hid dim]
        
        # Head ??? ?????? dim??? ????????? ??? head??? ???????????? query, key, value ??? ??????
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [batch size, n heads, query len, head dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [batch size, n heads, key len, head dim]
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [batch size, n heads, value len, head dim]
        
        # query??? key??? ????????? ?????? ??? ?????? ??? ??? ????????? ?????? scale????????? normalize ???
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # [batch size, n heads, query len, key len]
 
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # softmax??? ?????? normalize?????? attention score matrix??? ??????
        attention = torch.softmax(energy, dim = -1)   # [batch size, n heads, query len, key len]
        
        # attention score matrix??? value??? ?????? ??????
        x = torch.matmul(self.dropout(attention), V)  # [batch size, n heads, query len, head dim]
 
        x = x.permute(0, 2, 1, 3).contiguous()    # [batch size, query len, n heads, head dim]

        # ??? head??? concat?????? ?????? dimension?????? ??????
        x = x.view(batch_size, -1, self.hid_dim)  # [batch size, query len, hid dim]
        # W_O??? ????????? ?????? output??? ??????
        x = self.fc_o(x)  # [batch size, query len, hid dim]
   
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x): 
        # x : [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))  # [batch size, seq len, pf dim]
        x = self.fc_2(x)    # [batch size, seq len, hid dim]
 
        return x

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg : [batch size, trg len]
        # enc_src : [batch size, src len, hid dim]
        # trg_mask : [batch size, 1, trg len, trg len]
        # src_mask : [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device) # [batch size, trg len]

        # token embedding & positional encoding
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos)) # [batch size, trg len, hid dim]

        # decoder layers
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask) # [batch size, trg len, hid dim], [batch size, n heads, trg len, src len]
         
        output = self.fc_out(trg) # [batch size, trg len, output dim]
   
        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg : [batch size, trg len, hid dim]
        # enc_src : [batch size, src len, hid dim]
        # trg_mask : [batch size, 1, trg len, trg len]
        # src_mask : [batch size, 1, 1, src len]
        
        # Masked Multi-Head Self Attention : encoder??? self-attention??? ?????? ?????? ????????? masking
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg)) # [batch size, trg len, hid dim]
   
        # Multi-Head Cross Attention : Query??? decoder vector / Key-Value??? encoder vector
        # Encoder??? ????????? ???????????? ??????
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg)) # [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))  # [batch size, trg len, hid dim]
 
        # attention : [batch size, n heads, trg len, src len]
        
        return trg, attention

class Transformer(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        # Make attention masks
        # src : [batch size, src len]
        
        # <pad>?????? ?????? ?????? masking??? ?????? ??????
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # [batch size, 1, 1, src len]
        return src_mask
    
    def make_trg_mask(self, trg):
        # Make attention masks
        # trg : [batch size, trg len]
        
        # <pad>?????? ?????? ?????? masking??? ?????? ??????
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2) # [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]
        
        # ?????? ?????? ?????? ?????? token?????? ?????? ?????? masking
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool() # [trg len, trg len]

        # ??? mask??? ??????
        trg_mask = trg_pad_mask & trg_sub_mask # [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):
        # src : [batch size, src len]
        # trg : [batch size, trg len]
        
        # encoder??? attention mask??? decoder??? attention mask ??????
        src_mask = self.make_src_mask(src)  # [batch size, 1, 1, src len]
        trg_mask = self.make_trg_mask(trg)  # [batch size, 1, trg len, trg len]

        # Encoder
        enc_src = self.encoder(src, src_mask) # [batch size, src len, hid dim]
        # Decoder        
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)  # [batch size, trg len, output dim], [batch size, n heads, trg len, src len]

        return output, attention

# Hyper parameters
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)

# ignore_index
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
# Define model
model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

# weights initialize
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
model.apply(initialize_weights)

LEARNING_RATE = 0.0005
# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
# Define loss function
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg # [batch size, trg len]
        
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1]) # batch size, trg len - 1, output dim]
    
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim) # [batch size * trg len - 1, output dim]
        trg = trg[:,1:].contiguous().view(-1) # [batch size * trg len - 1]

        # loss ??????    
        loss = criterion(output, trg)
        # back propagation
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # update parameters
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:,:-1]) # [batch size, trg len - 1, output dim]

            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim) # [batch size * trg len - 1, output dim]
            trg = trg[:,1:].contiguous().view(-1) # [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

## Model Training~ ##
N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# Load trained weights & Test
model.load_state_dict(torch.load('tut6-model.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

## Inference ##
def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    # model evaluation mode
    model.eval()
    # source sentence tokenize
    if isinstance(sentence, str):
        nlp = spacy.load('de_core_news_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    # enocder??? attention mask ??????
    src_mask = model.make_src_mask(src_tensor) 
    # Encoder ??????
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    # decoder??? ??? ??????????????? <sos> ?????? ??????
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        # Decoder??? attention mask ??????
        trg_mask = model.make_trg_mask(trg_tensor)
        # Decoder ??????
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        # Prediction
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)
        # translation ??????
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attention

# Inference??? ????????? ????????? ?????? ??????
example_idx = 8
src = vars(train_data.examples[example_idx])['src']
trg = vars(train_data.examples[example_idx])['trg']

print(f'src = {src}')
print(f'trg = {trg}')

# ?????? ??????
translation, attention = translate_sentence(src, SRC, TRG, model, device)

print(f'predicted trg = {translation}')

# Attention matrix ?????????
def display_attention(sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):
    
    assert n_rows * n_cols == n_heads
    
    fig = plt.figure(figsize=(15,25))
    
    for i in range(n_heads):
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['']+[t.lower() for t in sentence]+[''], 
                           rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

display_attention(src, translation, attention)