# NanoGPT from Andrej Karpathy

This is what I built + my notes, following the NanoGPT guide by Andrej Karpathy. If you want to get some intuition on how GPT works, watch about 15 mins of [this video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2286s), starting at 42.13. Karpathy was an OpenAI co-founder and the Telsa autopilot vision team lead. Between Tesla and OpenAI, he released this gem of a guide.
​
Key Takeaways:
- Attention is a communication mechanism. Each token (sub-word) "searches" for information from the past. It says something like "I am a vowel, looking for consonants, in these positions"
- This happens in a data-dependant manner
- You can chop any data (images, sounds...) into chunks, mark the positions, and feed into the same architecture and it works
- GPT is a decoder model. It doesn't have memory or internal state. It only looks at a fixed context of input to generate outputs. GPT-4 Turbo takes in 128k input tokens and only looks at those.
​
The transformer architecture is revolutionary:
- It can "learn" during the inference time
- The architecture has not changed much in the past 6 years, since the 2017 "Attention is All You Need". Many are trying to improve it, but it has remained resilient.

## How to use

```
python building_gpt.py
```
will build the model and train on Tiny Shakespere dataset. Check the history of the file for step by step guide.

## Masked Self Attention

Attention(x) = V.σ(QK/sqrt(H))

|         | Info                                                                                                                             |
| ------- |--------------------------------------------------------------------------------------------------------------------------------- |
| x       | private information of token (B,T,C)                                                                                             |  
| q       | each token generates a query vector , [I am a vowel in position 8 looking for consonents upto position 4]                        |  
| k       | each other token generates a key vector, what information I have [I am a consonent in position 3]                                |
| w=qk    | affinity - those two tokens find each other, affinity at the intersection will be very high (I am interested in these positions) |
| v       | value vector, what information I am willing to provide                                                                           |
| y=vσ(w) | accumulate all the information from interested positions to me (B,T,H)                                                           |

```py
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

B = 4  # batch, independant sets processed in parallel for efficiency
T = 8  # time/block/context, window of tokens considered
C = 32 # in_channels, each token is "embedded" into a vector of size C
H = 16 # out_channels, seperate similar operations on same vectors

x = torch.randn(B,T,C)


key   = nn.Linear(C, H, bias=False)
query = nn.Linear(C, H, bias=False)
value = nn.Linear(C, H, bias=False)

# x          # (B,T,C)
k = key(x)   # (B,T,H)
q = query(x) # (B,T,H)
wei = q @ k.transpose(-2,-1) # (B,T,H) @ (B,H,T) = (B,T,T)

tril = torch.tril(torch.ones(T,T))               # for decoder: lower triangular matrix (including diagonal)
wei  = wei.masked_fill(tril==0, float('-inf'))   # Replace upper triangular of wei with -inf
wei /= H**0.5                                    # scaling, bring variance to 1, to prevent softmax clipping
wei  = F.softmax(wei, dim=-1)                    # -inf -> 0, rest normalized to 1

v = value(x)   # (B,T,H)
out = wei @ v  # (B,T,T) @ (B,T,T) = (B,T,H)
```

### Notes

Attention is a communication mechanism
- A directed graph of T nodes, each node being a token position, and contains info as a vector of H size
- T nodes aggregate infromation as a weighted sum from all nodes
- Data dependant, the data stored in the nodes change with time
- For autogression,
  - 1-th node gets only from itself
  - 2-nd node gets from 1,2
  - T-th node gets from everyone
- Nodes have no notion of space / ordering. So we need to add postional embedding

Encoder vs Decoder

- Encoder
  - no triangular mask (it gathers data from both directions)
  - eg. translation, sentiment analysis
- Decoder
  - triangular mask (gathers data from past & present only)
  - predicts next word
  - "autoregressive", P(next_word) = P(this_word|past_words) * P(prev_word|words_before)...

Attention = V*softmax(QK.T/sqrt(H))

- Self-Attention: Q,K,V come from X
- Cross-attention: 
  - query from x, keys & values come from different place. 
  - eg: English -> French, french (query) searches in English (key, value)


## Output from Model

```
RICASTASCAPULAUS:
StOndestrears on lyie.

DUKING grepheeld here frock rup:
Is 'twas uppeontized but din to him.

NORFirst Haltist:
The bay! I am as yon He litter.
This the more prumpetutalest? why proud you age!

CLARENCE:
Yet now, O, more storna.

FRORINZE

JULINA:
Provoker, your got, my horour lord:
Priveragin, goodip you sileng: Go thy frience more;
You must betteen be that great to shaltle an you see
By somey out of us: if your lords, which are it what you?

GLOUCESTER:
O thre paperswors I prouse here; what it was not fear the
is ust shalle, service, the noble's
Helm the duty?

BRUTUS:
O'tYou, let us Rome, we putill, for that God-morriestment
So thy attend-rigns, ear that more I do usa
Samep to malie roge. Clain, am she hermined.

ISAABELLA:
O mertaity of honess out act they take
For he farled As illest curity! That it 't;
For the to some, girious proop, to hoursest
Are the purled make: a king of my offend. York well
fight were it a train: I been and do moust thou be tus:
I know me alone too a proce, thy brother, thine
may the be not flowary bether in her teph this.

DUCHE:
Tybalt,
Here, Romeomeof, well, Who hark both a trongumes,
Grest morrow, and Clarences; to there theyS namest?
I Like betteeing him than up the lament:
Earlisant But I him speak it it a hot severe
Tyban which did is: agges-and yet? on's will I'll demass unhas
With ares, tastilly our rabred me; but let prithes yours.
Now never Sot me, now for have him
A caugatifit wars in clate? what is now you.
Say, you draying, I say! these how be
shell have winds and here, where a two fash one!
```