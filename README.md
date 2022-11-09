# NLP_research_interviews

## Generative AutoEncoder Models Comparison
### VAE vs Diffusion model

## Boosting

## Transformer-based models
### Attention vs LSTM
RNN is hard to parallelize because of its sequential nature. CNN can parallelize, but each filter/kernel can only see limited vectros. Filters from higher layer have potential to see more vectors. Self-attention layer has Q, K, V and token at any position can attend other tokens at any positions. Self-attention has similar idea with CNN, but has more parameters and therefore is more flexible and needs more training data, e.g. ViT has 300M training pictures, but if only trained on ImageNet, its performance is worse than CNN. 

### Self-attention
The core idea of attention is to compute Q, K, V, which is imported from recommondation system. Q represents the aspects we are interested in, K represents the aspects that the data have, while V represents the actual information of different aspects. The output of a self-attention layer is just a weighted average of the channels of the input features. By default, e.g. BERT, Q, K, V should have the same dimensions. However, this is not necessary. The attended output should have less dimension than the input features. For example, if K and V have dimension [T+R, d], while Q has [R, d] (because we are only interested in less channel/features than the input.), then the score metrics F(Q, K) has dimension [R, T+R], which behaves more like a feature selection/projection matric. The the atteneded output is [R, d], which has the same dimension with Q.

## NLG Metrics
- BLEU
- ROUGE

## Batch Normalization 

## SGD
