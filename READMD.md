# LoRA-Finetuned Sentiment Analysis

# Description

This application uses LoRA to fine-tune DistilBERT for sentiment analysis on the IMDB dataset. LoRA facilitates efficient adaptation of pre-trained weights while preserving model expressiveness. Additionally, it can do this without causing delays during inference.

# LoRA(Low-RANK ADAPTION of LARGE  LANGUAGE MODELS)

## Problem of directly train on pre-trained model

When fine-tuning on the pre-trained model, it creates a new model with a set of weights which have the same dimension and size as the pre-trained model. Storing and deploying multiple instances could be challenging, especially in edge computing.

Furthermore, even with training parts of the layers of the pre-trained model, the dimension of the layer is typically large, especially in modern LLM (Large Language Model). This introduces difficulty in training, both in terms of time consumed and the size of VRAM required.

## Problem of existing Solution

Many efforts have been made to solve this problem. By adding an adapter layer after original layer, it could lower the trainable variable and reduce the size of parameters that need to be saved. However, even with the smaller trainable parameter, it inevitably introduces latency due to the extra computation in the adapter layer. The reason that the latency is inevitable is that the adapter layer has to be processed sequentially, limiting the FLOPs (floating-point operations per second), and there is no way to bypass them. Even though the latency could be reduced by parallel computing, during inference, the batch size is usually small, resulting in difficulty reducing latency with parallel computing.

## LoRA

LoRA achieves lower GPU memory usage, faster training and no additional inference latency by injecting the lower-rank decomposition metric and freeze the original layer.

The equation of original model is $\text{output} = W_0x + b_0$ where $W_0 , b_0$ are the weights from from original fully connected layer and $W_0 \in R^{d\times k}$, which would be frozen while training with LoRA. The equation of LoRA is $\text{output} = W_0x + b_0 + BA x$, where A and B are decomposition metric and $B\in R^{d\times r}$, $A \in R^{k\times r}$, and the rank $r \ll min(d, k)$.  Which could greatly reduce the trainable parameter than directly update the weight of the layer( e.g. assume $d, k = 768$ and $r = 4$, it would be $768 \times 4  + 4  \times 768 = 6144$ comparing to original layer 589824). For the initial weight of the A, the Gaussian initialization is used in paper but Kaiming/He Initialization is used in practice, and zero for $B$. This mean at the beginning of the training, $\Delta W = BA = 0$. It scale the $\Delta W$  by $\frac{\alpha}{r}$, where $\alpha$  is a constant to decide how strongly LoRA affect to original layer.

Given that the new output $\text{output} = W_0x + BAx + b_0$, it is equals to $\text{output} = (W_0 + BA)x +b_0$. And since $W_0$ and $BA$ have the same shape of $d\times k$, it could added to the original weight, which solve the problem of the latency while inference. And to further train the model, the original weight could be restore by subtracting $BA$.

To apply LoRA to the transformer, the researchers develop the LoRa only apply in the attention weight for simplicity, and upon the projection of query, value, keys, and output projection, it works the best in only applying to query and value.

# Implementation of LoRA layer on DistilBert

## Create LoRA layer

In this project, the DistilBert from transformer created by Hugging Face is used. Rather than using Tensorflow’s MultiHeadAttention class, it creates its own MultiHeadAttention class. And one of the difference is that in Tensorflow, it create the weights with the size of $\text{heads} \times \frac{\text{dim}}{\text{head}}$ ; while Hugging Face first create the weights with size of $\text{dim}$, then reshape the output of it to $\frac{\text{dim}}{\text{head}}$.  So in this application, the $A$, $B$ in LoRA are dense layer with the shape of $\text{input dim} \times r$ and $r \times \text{input dim}$ respectively. 

## Injecting into model

The design of the transformers package put the transformer block in the distilbert.transformer.layer, so the injection could be done by looping through the Transformer block and replace the q_lin and v_lin with LoRA in its attention layer.