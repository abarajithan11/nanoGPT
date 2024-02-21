import torch
import torch.nn as nn
from torch.nn import functional as F

'''
------------------ Hyperparameters ----------------------------
'''
batch_size = 32 # B: how many independent sequences will we process in parallel?
block_size = 8  # T: what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
torch.manual_seed(1337)

'''
------------------ Data Loading ----------------------------
'''
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

chars_str = ''.join(chars)
print(f'vocab_size: {vocab_size}')
print(f'vocabulary: {chars_str}')

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


'''
------------------ Model ----------------------------
'''

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # for every possible token, weights for next token
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):

        tok_emb = self.token_embedding_table(idx)                                        # (B,T,C=n_embed)
        pos_emb = self.position_embedding_table(torch.arange(block_size, device=device)) # (T,C): [0,1,2..T-1]

        '''
        B - batch               # of independant vectors processed
        T - time/block/context  # of tokens in a context
        C - vocab               # of possible tokens
        '''

        x = tok_emb + pos_emb     # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):                        # idx is (B, T) array of indices in the current context
            idx_cond = idx[:, -block_size:]                    # crop the last block_size tokens for input
            logits, loss = self(idx_cond)                      # get the predictions
            logits = logits[:, -1, :]                          # (B,T,C) -> (B, C)
            probs = F.softmax(logits, dim=-1)                  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # sample from the distribution acc to prob (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)            # New idx is concat (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)


'''
------------------ Training ----------------------------
'''
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:   # every once in a while evaluate the loss on train and val sets
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train')     # sample a batch of data

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


'''
------------------ Generate Output ----------------------------
'''

context = torch.zeros((1, block_size), dtype=torch.long, device=device)  # start with '\n\n\n\n' as seed
out_ints = m.generate(context, max_new_tokens=500)[0].tolist() # output list of ints
print(decode(out_ints))