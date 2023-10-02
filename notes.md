# Learnings from 'makemore'

## Part 1: https://www.youtube.com/watch?v=PaCmpygFfXo

- Language models encode/preprocess training data by converting from a list of characters to a list of integers, where each integer represents a 'token'. A token can be as small as a single character and as large as a whole word, depending on the tokenization scheme used.
- Similarly, outputs of the language model must be decoded to return to a textual format.
- In this example we use character-level tokens, though for real-world models like GPT, more complex sub-work tokenizers are used.
- A bigram is a pair of adjacent characters. e.g. for the word "spork", you can make the following 6 bigrams: `[".s", "sp", "po", "or", "rk", "k."]` (where `.` represents the start/end of a word).
- An example of a simple language model is a Bigram language model. It models the probabilities of a character following another character, for all characters in the alphabet of a given training text.
- Because it uses only the preceding character to predict the next character, we say it has a "context size" of 1.
- This can be done statistically by storing normalised counts of occurances (i.e. for 26+1 length alphabet, storing a 27x27 matrix of probabilities (or 27 separate probability distributions)).
- Text can be generated from this model by simple sampling from this matrix (starting with the start/end char).
- A basic NN approach is to have:
  - a single char as input
  - one-hot, then matrix multiply by weights (i.e. linear layer, size 27x27) to give a 27-length vector of logits (like counts?)
    - Note: this series of operations (one-hot -> MM by weights) is what the torch.nn.Embedding operation does. i.e. the embedding operation is a lookup table of the weights using the input values/indices.
    - This kind of operation is referred to as a 'gather'.
    - During backpropagation, the weights are updated at the particular indicies via a scatter operation.
  - target is a one-hotted vector of the index of a single char
  - calculate the NLL loss based on softmaxed linear output (to get probs) vs target
- The bigram model is not very powerful as a language model, as it only takes a single character of context when predicting the next character.
- But if we want to use more characters of context, we see that the approach of tracking the distribution via counting is not scalable, as the probability matrix we have to store scales with the power of the context size (e.g. for 2-character context, we need a (27*27)x27 probability matrix)

## Part 2: https://www.youtube.com/watch?v=kCc8FmEb1nY

- Generally, the embedding tensor is a 2D tensor of shape (VocabSize, EmbeddingSize).
- In the bigarm model example, we deliberately designed the model such that embedding a token was equivalent to looking up the probabilities of the next token. So EmbeddingSize was equal to VocabSize (one probability for each possible token).
- However, the embedding of a token can be thougth of more generally as a feature vector for that token that:
  1. encodes semantic relationships - tokens with similar meaning have similar embeddings
  2. can be combined with context-specific information for greater model power (self-attention, explained later)
- So the EmbeddingSize can be chosen arbitrarily and isn't necessarily linked to vocab size.
- In that case we can use a linear layer to convert the token embeddings (outer dimension `C`) to logits (outer dimension `V`) that can be softmaxed to get probabilities, from which we can calculate a loss as before.
- In reality we pass the gathered token embeddings for the input sequence to a bunch of (self-attention + feedfordward, spoiler alert!) layers before finally calculating the logits.

- Take the following sequence of tokens: [0, 13, 72, 4, 99, 92]
- So far we have only looked at using the embeddings for the previous token to predict the next token.
- e.g. in the bigram model, if we wanted to predict the token following `13`, we would look up its embedding, and sample from its multinomial distribution. But when we have a sequence of tokens, we can use the embeddings of the [0, ..., N-1] tokens when predicting the Nth token. This is what we do in the self-attention mechanism.
- Note that tokens can't 'talk to' to 'future' tokens.
- A simple (but very lossy!) way of context aggregation could be to take the mean the embeddings of the previous tokens.
- What this looks like in MM terms:

```python
x = torch.tensor([0, 13, 72, 4, 99, 92]) # input: a list of tokens of length 'context size', (T)
T = x.shape[0] # time dimension
V = int(x.max().item() + 1) # vocab size
C = 3 # embedding channels
embed = torch.nn.Embedding(V, C) # embedding, (V, C)
y = embed(x) # embeddings at indices in x, (T, C)
print(y)
"""
tensor([[-0.6410, -0.7645, -0.8557],
        [-0.1462, -0.0190,  0.5199],
        [-0.2555,  0.8289,  0.2897],
        [ 0.9589, -0.1687,  0.0422],
        [-0.6383, -1.5480, -0.9523],
        [-0.3593, -1.1079, -1.2822]], grad_fn=<EmbeddingBackward0>)
"""

# The expected result would look like
"""
[[-0.6410, -0.7645, -0.8557],
 ([-0.6410, -0.7645, -0.8557] + [-0.1462, -0.0190,  0.5199]) / 2,
 ([-0.6410, -0.7645, -0.8557] + [-0.1462, -0.0190,  0.5199] + [-0.2555,  0.8289,  0.2897]) / 3,
 ...
]
"""

# This can be achieved in MM by using a normalized triangular matrix:
w = torch.tril(torch.ones(T, T))
w = w / torch.sum(w, 1, keepdim=True)
print(w)
"""
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]])
"""
aggregated_y = w @ y

# Note you can also get `z` by doing a row-wise softmax of a zeros tensor with
# `-inf`s in the upper right triangle:
w = torch.zeros(T, T)
w = w.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
print(w)
w = torch.nn.functional.softmax(w, dim=-1)
assert torch.allclose(w @ y == aggregated_y)
```

The matrix `w` defines the 'Affinities' - how much do we want to weight the embedding of each preceding token when aggregating the result for a give token. These are the weightings of the aggregation function. The `-inf` values indicate that we want to ignore all 'future' tokens.

So now we have a way of considering all previous tokens within the context when predicting the next token. But we're missing two ingredients to get to self-attention:

1. A way of weighting the embeddings of tokens based on their position in the context
2. A way of making this weighting data-dependent

Here set the values of the affinity matrix statically to always take a simple average of preceding token embeddings, but in self-attention we calculate these affinities dynamically to achieve these objectives.

### Encoding position in the sequence embedding

To also encode the position of the tokens in the tensor that we pass to the first head of the network (see (1) above), we introduce a second embedding matrix: a position embedding matrix. This can be combined, via simple addition, with the gathered embedding vectors for the tokens in the input sequence:

```python
token_embed = torch.nn.Embedding(V, C) # token embedding table, (V, C)
pos_embed = torch.nn.Embedding(T, C) # pos embedding table, (T, C)
t_embed = token_embed(x) # embeddings at indices in input, x, (T, C)
p_embed = pos_embed(torch.arange(T)) # == pos_embed.weight, (T, C)
y = t_embed + p_embed
```

We will see below that the self-attention mechanism doesn't have a notion of position, when something about one token is communicated to another. It is only from adding this positional embedding that this information is encoded (and then communicated in self-attention).

### Self-attention: "learning to pick affinities data-dependently"

Each token emits 3 vectors:

1. A query: "what am I looking for?"
2. A key: "what do I contain?"
3. A value: "what I want to communicate to other tokens"

Affinities (see the above `w` matrix`) between tokens in a sequence are calculated by doing dot-products between Queries and Keys. "How much does the key for each token match with the queries of the other tokens"?

Each token's Query vector dot-products with the Keys of all other tokens, and these results form the aggregation weightings of `w`.

Let's see what this looks like:

```python
# Initialise an activation tensor, either the token embeddings
# (remember y = embed(x)) or an intermediate activation in the model that has
# the same shape.
T, C = 3, 5 # context size, embedding channels
y = torch.randn(T, C)
head_size = 16 # Some hyperparameter of the model, H
get_key = torch.nn.Linear(C, head_size, bias=False) # get_key.weight shape (H, C)
get_query = torch.nn.Linear(C, head_size, bias=False) # get_query.weight shape (H, C)
# (T, C) @ (C, H) matrix multiplications
k = get_key(y) # (T, H)
q = get_query(y) # (T, H)
```

The matrix multiplies act independently across the context size (`T`) dimension, so `k` and `q` are the idependent key and query vectors corresponding to each token in input `x`.

Now we dot-product each query with each key (see `q @ k.T`). This result initializes the affinities in our weighted aggregation matrix `w`. We still zero the top right triangle (as tokens can't see into the future), and normalize:

```python
w = q @ k.T # == k @ q.T, (T, T)
w = w * head_size ** -0.5
w = w.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
w = torch.nn.functional.softmax(w, dim=-1)
```

Note we normalize weights by `1/sqrt(head_size)` to maintain unit variance at initialization. Otherwise the softmax operation will result in a more 'pointy' output (largest value in input will map to an even larger value in output relative to other values).

Now the aggregation weightings/affinities/attention scores are also data-dependent (see (2) above).

Let's tell a story of the experience of a single token as it travels through the first attention head to demonstrate the above concepts in action:

- A sequence of tokens is passed into the network, `[_, _, _, M]`.
- The last token in the sequence, `M`, is represented as an index into the vocabulary of the input data set.
- The token's value is used to gather an embedding vector representing that token `y_M = token_embed(onehot(M))`.
- This gives the token `M` semantic value: we now have some better information about what `M` _is_.
- Then we add this vector to the positional embedding vector for the last token in the context `y_M += pos_embed(onehot(T-1))`.
- Now this vector contains semantic information about the token, as well as information about the token's position in the sequence. e.g. "I'm the letter 'a', and I'm in the last position in the context"
- Now we move this vector/embedding along, into the first attention head. We do two matrix multiplications to compute the key and query for this embedding, based on its semantic and positional knowledge of itself.
- e.g. the query could be "I'm looking for consonants in positions 0-3", and the key could be "I'm a vowel, and I'm in the second half of the context".
- All other tokens emit keys (and queries) as well. We use these to compute the affinity matrix. Let's take a look at the final row, `w`, corresponding to the affinities with the final token in the context:
  - `print(w[-1]) -> [0.01, 0.55, 0.16, 0.28]`. Here we see that the second token has a high affinity (0.55) with our token `M`, meaning its key matches well with the `M`'s query.
- So when we perform the aggregation `softmax(mask(w)) @ y`, we end up incorporating lots of the embedding from the second token into the output vector corresponding to `M`'s token.

Unlike in the simple example above, we do not MM the affinities directly with the input embeddings for the sequence, but with the Values corresponding to these tokens. Attention heads aggregate Values (linearly transformed embeddings corresponding to input tokens) based on how well the Keys match with the Queries.

```python
get_value = torch.nn.Linear(C, head_size, bias=False) # get_value.weight shape (H, C)
v = get_value(y) # (T, H)
aggregated_y = w @ v
```

### Self-attention as a communication mechanism

AK uses the analogy that self-attention is like communication between nodes in a directed graph. The nodes are the tokens in a sequence, and the communication pattern is defined by the constraint that nodes can only speak with previous nodes/tokens in the context. The information that is communicated along the edges of the graph is data-dependent.

Note that this constraint that 'tokens can't talk to future tokens' that we make with the zeroing of the upper triangle of `w`:

```python
w.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
```

doesn't have to be applied in all language models. We do it here because we're trying to predict the next character. But if we wanted to do e.g. sentiment analysis/text classification, we wouldn't necessarily apply this constraint.

AK says that the upper triangular zero-mask is applied to attention heads in decoder blocks, but not in encoder blocks.

Note we call these GPT models "autoregressive":

> "autoregressive" refers to the property of generating outputs one step at a time in a sequential manner, where each step depends on the previously generated steps.

### The 'self' in self-attention

We call it self-attention because the queries, keys and values all come from the same source (in the above code snippets, the input `y`). Cross-attention is where the keys and values come from a separate source.

If you look at the diagram from the 'attention is all you need' paper, there's cross-attention in a decoder layer, as:

- the Queries are still emitted by the input `y`
- but the Keys and Values are emitted from the output of an encoder block.

This is because the model proposed in the paper is used for language translation. So the encoder is used to encode the 'sentence to be translated' (without using the triangular mask, as all tokens can speak to eachother), and the `k,v`s are passed to the decoder blocks, which generate the translated sentence tokens.

Decoder-only networks (including GPTs) do not contain cross-attention layers (as they do not need to condition on the output of any encoder layers).

### Encoder vs Decoder

The model described in the AIAYN paper is an 'encoder-decoder' model. The only difference between the two in the self-attention layer is just whether or not the attention weights (`w`) is zero-ed out in the top-right triangle. If not, this allows tokens to communicate with later tokens in the sequence. This is allowed in encoder layers, but not in decoder layers.

The paper used attention in a machine language translation application, where the sentence to be translated was processed by the encoder, then the translated sentence was generated by the decoder.

BERT is an example of a 'encoder-only' model, because all its attention layers allow for communication between all tokens in the input sequence. That's because its downstream applications are e.g. text classification, and sentiment analysis where a prediction is made based on all tokens at once.

GPTs are examples of 'decoder-only' models because they only allow tokens to communicate with themselves and ealier tokens, and then auto-regressively predict the next token.

### Multi-head attention

- Multiple self-attention heads act on the input tensor in parallel.
- Their outputs are concatented along the channel axis, then passed through a linear layer, so the outputs of individual heads are combined.

### Feedforward layer

- After the attention layer (in actual paper, 1x self attention if encoder block, or 1x SA followed by 1x cross attention if decoder block), there is a feed-forward layer.
- AK gives the analogy of "the self-attention layer allows tokens to communicate with eachother different aspects of what that contain (message passing of features), whereas the FF layer allows the tokens to think about the messages they were passed".

### Optimisation tricks

To make the network more trainable, the AIAYN paper employs typical DL model tricks:

- Skip connections, to prevent vanishing gradients during backprop. Important for deep networks.
- Layernorm to improve stability of gradients during training. Apparently the location of norms has changed in standard practice vs. what is originally proposed in the paper.
- Dropout layers at various points in the block, to prevent overfitting.
