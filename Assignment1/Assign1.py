import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

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
Downloading
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
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
The field appends the "start of sequence" and "end of sequence" tokens via the init_token and eos_token arguments, and converts all words to lowercase.
"""
# Source field
SRC = Field(tokenize = tokenize_de, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)
# Target field
TRG = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

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

# Building the Seq2Seq Model - Encoder
"""
Bidirectional RNN : the annotation of each word to summarize not only the preceding words, but also the following words
z = s0 (hidden) : concatenating two context vectors (a forward and a backward one) together => initial hidden state of decoder
"""
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src : [src len, batch size]
        
        # Embedding the input src
        embedded = self.dropout(self.embedding(src)) # [src len, batch size, emb dim]
        
        # Perform Bidirectional RNN
        outputs, hidden = self.rnn(embedded) # [src len, batch size, hid dim * num directions] , [n layers * num directions, batch size, hid dim]
        
        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer        
        # hidden [-2, :, : ] is the last of the forwards RNN 
        # hidden [-1, :, : ] is the last of the backwards RNN
        
        # initial decoder hidden is final hidden state of the forwards and backwards 
        # encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))) # [batch size, dec hid dim]
        
        # outputs : H = {h1, h2, ... , hT} , h1 = [h1->,h1<-] ...

        return outputs, hidden # [src len, batch size, hid dim * num directions], [batch size, dec hid dim]

# Attention mechanism
"""
Inputs : the previous hidden state of the decoder s_(t-1) and all of the stacked hidden states from encoder H.
Outputs : an attention vector a_t that is the length of the source sentence.
          Each element is between 0 and 1 and the entire vector sums to 1.

a_t represents which words in the source sentence we should pay the most attention to 
in order to correctly predict the next word to decode y_(t+1).
"""
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):       
        # hidden (z) : [batch size, dec hid dim]
        # encoder_outputs : [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        # repeat decoder hidden state src_len times
        # encoder_outputs와 shape를 맞춰주기 위함
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) # [batch size, src len, dec hid dim] s0
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # [batch size, src len, enc hid dim * 2]
            
        # Calculate energy between the previous decoder hidden state and the encoder hidden states    
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) # [batch size, src len, dec hid dim]
        
        attention = self.v(energy).squeeze(2) # [batch size, src len]
                
        return F.softmax(attention, dim=1)

# Building the Seq2Seq Model - Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):             
        # input : [batch size]
        # hidden : [batch size, dec hid dim]
        # encoder_outputs : [src len, batch size, enc hid dim * 2]
               
        input = input.unsqueeze(0) # [1, batch size]
                
        embedded = self.dropout(self.embedding(input)) # [1, batch size, emb dim]

        # Get an attention vector     
        a = self.attention(hidden, encoder_outputs) # [batch size, src len]        
        a = a.unsqueeze(1) # [batch size, 1, src len]
                
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # [batch size, src len, enc hid dim * 2]
        
        # Matrix 연산을 통해 weighted source vector w_i를 구함. context vector c_i와 같음
        weighted = torch.bmm(a, encoder_outputs) # [batch size, 1, enc hid dim * 2]       
        weighted = weighted.permute(1, 0, 2) # [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2) # [1, batch size, (enc hid dim * 2) + emb dim]

        # 현재 hidden state s_i 구함            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0)) 
        # [seq len, batch size, dec hid dim * n directions], [n layers * n directions, batch size, dec hid dim]       
        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output : [1, batch size, dec hid dim]
        # hidden : [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0) # d(y_(i-1))
        output = output.squeeze(0) # s_i
        weighted = weighted.squeeze(0) # w_i = c_i
        
        # 다음에 올 단어 y_i 예측 
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1)) # [batch size, output dim]
                
        return prediction, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):      
        # src : [src len, batch size]
        # trg : [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
                
        # first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs

# Hyper parameters
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# Define model
attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

# weights initialize
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
model.apply(init_weights)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# ignore_index
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

# Define loss function
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX).to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters())

# Define training loop
def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # trg : [trg len, batch size]
        # output : [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        # trg : [(trg len - 1) * batch size]
        # output : [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# Define evaluation loop
def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) # turn off teacher forcing
            
            # trg : [trg len, batch size]
            # output : [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg : [(trg len - 1) * batch size]
            # output : [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# Model training
N_EPOCHS = 20
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
        torch.save(model.state_dict(), 'tut3-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# Evaluate
model.load_state_dict(torch.load('tut3-model.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
