import torch
import torch.nn as nn
import numpy as np

import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# Padding Should be Zero
src_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4, 'cola' : 5}
enc_vocab_size = len(src_vocab)

tgt_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'coke' : 5, 'S' : 6, 'E' : 7, '.' : 8}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
dec_vocab_size = len(tgt_vocab)

def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
      enc_input = [[src_vocab[n] for n in sentences[i][0].split()]] # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
      dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]] # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
      dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]] # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]

      enc_inputs.extend(enc_input)
      dec_inputs.extend(dec_input)
      dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

class MyDataSet(Data.Dataset):
  def __init__(self, enc_inputs, dec_inputs, dec_outputs):
    super(MyDataSet, self).__init__()
    self.enc_inputs = enc_inputs
    self.dec_inputs = dec_inputs
    self.dec_outputs = dec_outputs
  
  def __len__(self):
    return self.enc_inputs.shape[0]
  
  def __getitem__(self, idx):
    return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

d_model = 512
n_head = 8
n_layers = 6
d_k = d_v = 64

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.lm_head = nn.Linear(d_model,dec_vocab_size)
    
    def forward(self,encoder_input,decoder_input):
        # *_input (B,L)
        encoder_output, encoder_attns = self.encoder(encoder_input) # (B, L, d_model)
        decoder_output, decoder_attns = self.decoder(decoder_input, encoder_input, encoder_output) # (B, L, d_model)
        logits = self.lm_head(decoder_output) # (B, L, vocab_size)
        return logits

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.dec_emb = nn.Embedding(dec_vocab_size,d_model)
        self.position = PositionalEncoding()
        self.layers = nn.ModuleList([ DecoderBlock() for _ in range(n_layers) ])
    
    def forward(self,decoder_input, encoder_input, encoder_output):
        decoder_output = self.position(self.dec_emb(decoder_input))

        decoder_pad_mask = get_pad_mask(decoder_input,decoder_input)
        decoder_seq_mask = get_seq_mask(decoder_input)
        dec_mask = torch.gt((decoder_pad_mask + decoder_seq_mask), 0)
        dec_enc_mask = get_pad_mask(decoder_input, encoder_input)

        dec_enc_attns = []
        for layer in self.layers:
            decoder_output, dec_enc_attn = layer(decoder_output,encoder_output,dec_mask,dec_enc_mask)
            dec_enc_attns.append(dec_enc_attn)
        
        return decoder_output, dec_enc_attns

class DecoderBlock(nn.Module):
    def __init__(self):
        super(DecoderBlock,self).__init__()
        self.casual_mha = MultiHeadAttention()
        self.dec_enc_mha = MultiHeadAttention()
        self.ffn = FeedForwardNet()
    
    def forward(self,decoder_input,encoder_output,dec_mask,dec_enc_mask):
        dec_self_output, dec_self_attn = self.casual_mha(decoder_input,decoder_input,decoder_input,dec_mask)
        dec_enc_output, dec_enc_attn = self.dec_enc_mha(dec_self_output,encoder_output,encoder_output,dec_enc_mask)
        return self.ffn(dec_enc_output), dec_enc_attn


def get_seq_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).bool()
    return subsequence_mask


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.enc_emb = nn.Embedding(enc_vocab_size, d_model)
        self.position = PositionalEncoding()
        self.layers = nn.ModuleList([ EncoderBlock() for _ in range(n_layers) ])
    
    def forward(self,encoder_input):
        encoder_pad_mask = get_pad_mask(encoder_input,encoder_input)
        encoder_output =  self.position(self.enc_emb(encoder_input))
        encoder_attns = []
        for layer in self.layers:
            encoder_output, encoder_attn = layer(encoder_output,encoder_pad_mask)
            encoder_attns.append(encoder_attn)
        return encoder_output, encoder_attns

class EncoderBlock(nn.Module):
    def __init__(self):
        super(EncoderBlock,self).__init__()
        self.mha = MultiHeadAttention()
        self.ffn = FeedForwardNet()
    
    def forward(self,encoder_input,encoder_pad_mask):
        encoder_output, attn_score =  self.mha(encoder_input,encoder_input,encoder_input,encoder_pad_mask)
        return self.ffn(encoder_output), attn_score

class FeedForwardNet(nn.Module):
    def __init__(self):
        super(FeedForwardNet,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model,4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model,d_model)
        )
        self.layernorm = nn.LayerNorm(d_model)
    def forward(self,x):
        return self.layernorm(self.fc(x)+x)


class MultiHeadAttention(nn.Module):
    def __init__(self,):
        super(MultiHeadAttention,self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_head, bias=False)
        self.fc = nn.Linear(d_v * n_head, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)
    
    def forward(self,query,key,value,mask):
        residual, bsz = query, query.size(0)
        q = self.W_Q(query).view(bsz,-1 ,n_head, d_k).transpose(1,2)
        k = self.W_K(key).view(bsz,-1 , n_head, d_k).transpose(1,2)
        v = self.W_V(value).view(bsz,-1 ,n_head, d_v ).transpose(1,2)

        attn_weight = q @ k.transpose(-1,-2) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, n_head, 1, 1)
            attn_weight = attn_weight.masked_fill(mask,float("-inf"))
        
        attn_score = torch.softmax(attn_weight,dim=-1)
        
        context = attn_score @ v

        context = context.transpose(1,2).reshape(bsz,-1,n_head*d_v)
        output = self.layernorm(self.fc(context) + residual)
        return output, attn_score
        

def get_pad_mask(seq_q,seq_k):
    b, l_q = seq_q.size()
    b, l_k = seq_k.size()
    mask = seq_k.eq(0).unsqueeze(1)
    return mask.expand(b,l_q,l_k)
    

class PositionalEncoding(nn.Module):
    def __init__(self,max_len=100):
        super(PositionalEncoding,self).__init__()
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) # max_len, d_model
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self,input):
        # input (B,L,d_model)
        return input + self.pe[:,:input.size(1), :]
        

model = Transformer()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)


for epoch in range(30):
    for enc_inputs, dec_inputs, dec_outputs in loader:
      '''
      enc_inputs: [batch_size, src_len]
      dec_inputs: [batch_size, tgt_len]
      dec_outputs: [batch_size, tgt_len]
      '''
      # outputs: [batch_size * tgt_len, tgt_vocab_size]
      outputs = model(enc_inputs, dec_inputs)
      outputs = outputs.view(-1, outputs.size(-1))
      loss = criterion(outputs, dec_outputs.view(-1))
      print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:         
        dec_input = torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype)],-1)
        dec_outputs, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.lm_head(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab["."]:
            terminal = True
        print(next_word)            
    return dec_input

# Test
enc_inputs, _, _ = next(iter(loader))
for i in range(len(enc_inputs)):
    greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab["S"])
    predict = model(enc_inputs[i].view(1, -1), greedy_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict.squeeze()])