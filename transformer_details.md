# Transformer structure
Transformer has a encoder and decoder, each consists of 6 blocks.
- The output of the encoder is used to attended with every layers of the decoder.  
- Word embedding (e.g. random initialized, Glove) + sinodal positional embedding
- Multi-head self-attention: 8 heads, the results of each head will be concatenated and passed to next layer.
- Add & Norm sub-layer: **Add** is a residule connection to solve training problem of deep neural network such that the NN only needs to focus on the current residules. (gradient vanish, very commonly used in ResNet.) **Norm** stands for layer normalization (commonly used in RNN), which normalize the inputs of each layer into N(0,1) to speed up the convergence.        
- Feed Forward sub-layer: Two fully connected layers, the first activation is Relu and the second doesn't have an activation.

## Transformer encoder
- Each encoder block consists of a multi-head self-attention layer (with Add & Norm) and a Feed Forward layer (with Add & Norm)
- The output of the encoder is the encoded tokens which has the dimension of [# tokens, latten state] 
## Transformer decoder
- Each decoder block consiste of a 1) **masked** multi-head self-attention layer (with Add & Norm), 2) a multi-head cross-attention layer (with Add & Norm) and 3) a Feed Forward layer (with Add & Norm).
- The first masked multi-head self-attention: **The decoder mask will prevent the current token attending with the tokens behind it**.  
- The second multi-head cross-attention: **The K and V of this layer is calculated from the encoded input (output of the encoder), while Q is calcualted based on the previous masked attention layer (already masked)**. Each token in the decoder could attend with all tokens in the encoder. 
- 
## Encoder-decoder
