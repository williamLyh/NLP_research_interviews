# Transformer structure
Transformer has a encoder and decoder, each consists of 6 blocks.
- **The output of the encoder is used to attended with every layers of the decoder.**  
- Word embedding (e.g. random initialized, Glove) + sinodal positional embedding
- Multi-head self-attention: 8 heads, the results of each head will be concatenated and passed to next layer. Why multi-head instead of one: Multi-head forms multiple sub-space. Each sub-space can learn focus on different aspect of information, while the total attention size remain the same.   
- Add & Norm sub-layer: **Add** is a **residule connection** to solve training problem of deep neural network such that the NN only needs to focus on the current residules. (to tackle network degradation and gradient vanish, as used in ResNet.) **Norm** stands for **layer normalization** (commonly used in RNN), which normalize the inputs of each layer into N(0,1) to speed up the convergence.        
- Feed Forward sub-layer: **Two** fully connected layers, the **first activation is Relu** and the second **doesn't** have an activation.

## Transformer encoder
- Each encoder block consists of a multi-head self-attention layer (with Add & Norm) and a Feed Forward layer (with Add & Norm)
- The output of the encoder is the encoded tokens which has the dimension of [# tokens, latten state] 
### BERT
- Two pretrained objectives: Masked LM objective and Next Sentence Prediction objective.
- BERT input and output will have space for [CLS] token at the beginning of sequence and [SEP] token at the end of the first sentence. 
- Why [CLS] token can be used to represent the full sentence? Because there is no actual input word (no semantics) corresponding to [CLS] token, therefore, after 12 layer self-attentions, it has **value average the representation of all words after attention**. Furthermore, NSP pretraining objective is calculated from [CLS] token, therefore it can learn better sentence-level semantic information.
- BERT embeddings: The sum of token embedding, segment embedding and position embedding.

## Transformer decoder
- Each decoder block consiste of a 1) **masked** multi-head self-attention layer (with Add & Norm), 2) a multi-head cross-attention layer (with Add & Norm) and 3) a Feed Forward layer (with Add & Norm).
- The first masked multi-head self-attention: **The decoder mask will prevent the current token attending with the tokens behind it**.  
- The second multi-head cross-attention: **The K and V of this layer is calculated from the encoded input (output of the encoder), while Q is calcualted based on the previous masked attention layer (already masked)**. Each token in the decoder could attend with all tokens in the encoder. 

## Encoder-decoder
- The architecture is used for seq2seq problem. Encoder and decoder could have different options such as RNN, CNN or transformer.
- The encoder can also used to encode multimodal data and the decoder can generate sequence of different modality.
- Limitations: Traditionally, the only connection between encoder and decoder is **a fixed length semantic vector**. There might be **information loss**, especially when the input sequence is long.
- The **attention mechanism** in the transformer does **not require compressing information** into a fixed length vector.

## Self-attention
The core idea of attention is to compute Q, K, V, which is imported from recommondation system. Q represents the aspects we are interested in, K represents the aspects that the data have, while V represents the actual information of different aspects. The output of a self-attention layer is just a weighted average of the channels of the input features. By default, e.g. BERT, Q, K, V should have the same dimensions. However, this is not necessary. The attended output should have less dimension than the input features. For example, if K and V have dimension [T+R, d], while Q has [R, d] (because we are only interested in less channel/features than the input.), then the score metrics F(Q, K) has dimension [R, T+R], which behaves more like a feature selection/projection matric. The the atteneded output is [R, d], which has the same dimension with Q.
