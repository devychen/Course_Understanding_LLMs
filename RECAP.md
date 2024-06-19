## Learning goals:
**know about LM architectures, training and inference methods**
- be able to implement and train small LMs
- be able to implement different decoding methods for trained LMs
- understand the workings of fine-tuning and LM agents 

**increase skill for using pretrained LMs for practical applications**
- developed sharper intuitions about in-context learning / prompt engineering
- acquire an acute sense of critical distrust based on knowledge of LM architectures

**become familiar with the current literature on assessment & interpretability**
- know about state-of-the-art results in targeted assessment and machine psychology
- be able to implement simple attribution and probing methods
- understand how more sophisticated methods of mechanistic interpretability work 

**develop confidence to critically evaluate SOTA language technology and its implications** 
- build intuitions about implications for academia, education, industry, society
- anticipate potential risks of future language technology

## 1 Intro
- Begin with Machine Translation, now much broader: Name entity recognition, Tagging, Question Answering, Information Retrieval, Chatbots, etc...
![Image](/pics/history.png)

## 2 Pytorch
(skip)

## 2 Optimization (via Backpropagation) 
*调参降误* <br>
**How the optimisation for probabilistic model is achieved?** <br>
> In short:<br>
Compute predictions (get the current prediction) $\rightarrow$ compute the loss $\rightarrow$ backpropagate the error (in which direction to change) $\rightarrow$ update the parameters $\rightarrow$ zero the gradients (reset the info about direction to tune for the next training step) 

To optimise the model is to adject the model's parameters in order to minimise prediction errors during backpropagation steps. <br>
The process begins with defining a **loss function**, which quantifies the diffrence between the model's preduction values and actual target values.
This function provides a measure of how well the model is performing, with the goal being to minise the loss.
**Gradients** which are vectors of partial derivatives of the loss function with respect to the model parameters, are then computed. These gradients indicate the direction and rate of the change needed to reduce the loss.
**Backpropagation** is used to efficiently calculate these gradients by propagating the error backward through the network from the output to the input, updating each parameter along the way.
**Stochastic Gradient Descent (SGD)** is then employed to iteratively adjust the model's parameters.
Instead of using the entire dataset, SDG uses random subsets (mini-batches) of data to perform these updates, making the optimisation process more computatioanlly eifficient and allowing the model to converge more quicly to a minimum loss. Through repeated cycles of these steps - calculating the loss, determining the gradients, and updating the parameters via backpropagation and SDG - the probabilistic model becomes increasingly accurate in making predictions.

### loss function
> function to calculate the loss - the differences between model preduction values and actual target values
- Define: MSE $ = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$ or Cross-entropy loss = $-\frac{1}{n} \sum_{i=1}^n \sum_{j=1}^c y_{ij} \log(\hat{y}_{ij})$
- Compute: use model to generate predictions based on current parameters.
- Apply: calculate loss 
    - For MSE, compute the squared differences between predicted and actual values, then average them.
    - For Cross-Entropy Loss, compute the logarithm of the predicted probabilities, multiply by the actual class indicators, sum these values, and take the negative average.
- Aggregate: if using mini-batches in SGD, compute loss for each batch and average them to get overall.

### Gradient
> Vectors of partial derivatives of the loss function with respect to the model parameters. <br>
> It suggests the rate and direction to update the parameters (in the direction opposite to the gradient, $<0$ increase, $>0$ decrease) in order to minimise the loss function. 
- Define: specify the loss function first.
- Compute: compute the partial derivatives of the loss function with respect to each model parameter $\theta_j$​. The gradient is a vector of these partial derivatives.
- Update: use gradients to adject parameters - This is typically done using an optimization algorithm like Stochastic Gradient Descent (SGD).
- Iterate: repeat computing and updating for a number of iteration or until convergence

### Backpropagation
> A method (理解为*方法论*) used to compute gradients of a loss function <br>
> It leverages the chain rule of calculus to propagate the gradients backward through the network layers.

### Stochastic Gradient Descent (SGD)
> Optimization algorithm to minise the loss function by computing gradients iteratetively through mini-batches and update model parameters accordingly. <br>
> vs. **vanilla/Batch GD** - single trial on entire dataset
- Initialise: initialise model parameters
- Define: choose the appropriate loss function depending on task
- Set up: SGD optimiser, specifiying parameters like `lr` and `momentum`.
- Batch data: organise data into mini-batches
- Loop: iterate, forward and backward passes and updating parameters with SGD.

## 2 ANN - Artificial Neural Networks (esp. MLP)
> A specific type of neural network designed for artificial intelligence applications.
1. Components
- **Neurons** (nodes): basic unit. Each node receives ≥ 1 inputs, perfroms a computation (usually a wrighted sum of inputs), applies an acitivation function to the result, and then passes the ourput to the next layer.
- **Layers**: neurons are organised into layers.
    - Input layer: receive input data
    - Hidden layer: perform computations
    - Output layer: produce the final outputs
- Connections (**weights**): an associated weight that determines the strength and direction of influence from a neuron's input to its output. *Learnign in NNs are generally just adjusting these weights to improve performance.*
2. Types
- FFNN (Feedforward neural networks): info flows in one-direction: input-hidden-output
- RNN (Recurrent ~): designed to process sequential data where the output depends on the current & **past** input
- CNN (Convolutional ~): specilised for processing grid-like data (e.g. images, videos)

## 2 Language Models
> $LM$: A function that assigns to each input $X$ a probability distribution over $S$ given parameter $\theta \in \Theta$: <br>
i.e. $LM_A: X \mapsto \Delta(S)$ <br>
$X$: set of input conditions (images, texts in a different langauge) <br>
$S$: set of sequenes of tokens, also $W_{1:n} = \langle w_1, .., w_n \rangle$ <br>
$V$: vocabulary, set of tokens <br>
1. Causal LM (also left-to-right LM): 
    > a function that maps an initial token sequence to a next-token distribution: $LM: w_{1:n} \mapsto \Delta(V)$
    - next-token probability: $P_{LM}(w_{n+1}|w_{1:n})$
2. Training: 
    - minise next-word **Suprisal** - the uncertainty/unpredictbility of an event. Commonly measured by **Perplexity**
3. Prediction:
    - sample **Autoregressive generation**, using next-word probabilities
    > a method to generate sequences of data where each element is conditioned on previous elements.
    - the process: input word $\rightarrow$ embedding (covnerts semantic meanings to vector space) $\rightarrow$ RNN (maintains a hidden state that captures the context) $\rightarrow$ Softmax layer (final hidden state is passed through a softmax layer to produce a probability distribution over the vocab for the next word) $\rightarrow$ Sampled word (being chosen)
4. Evaluation: 
    - **Perplexity**: measure of goodness of fit, we should minise it
    - or Average surprisal (i.e. cross-entropy) - calculate: the log of perplextiy
5. Different training strategies:
    - **Teacher forcing**: input is true word sequence (ie. next word in training set) rather than model prediction.
    - Autoregressive training
    - curriculum learning/schedule sampling: hybrid of two above
    - professor forcing: combine teacher forcing with adversarial training
    - decoding-based: use prediction function (decoding scheme) to optimize based on actual output
        - pure, greedy, softmax, top-k, top-p sampling, & beam search
6. Comparisons
    - auto-regressive LM: one that only has access to the previous tokens (and outputs become inputs)
        - Evaluation: perplexity
    -  masked LLM: can peak ahed too, it masks a word within the context (e.g. the centre word)
        - Evaluation: downstream NLP tasks that use the learned embeddings

## 2 RNN
1. Training RNN: 
- Input embedding (convert words to vectors) $\rightarrow$ RNN Processing (update hidden states with each word) $\rightarrow$ Softmax over vocab (use final hidden state and softmax over vocabulary to compute next word probabilities) $\rightarrow$ Calculate loss (compare predicted next word with actual next word) $\rightarrow$ Backprogagate and update (backpropagate the error through the network and update parameters)
![image](/pics/RNN.png)
2. Mind the dimensionality
3. Strength: 
    - Can handle infinite-length sequences (vs. just a fixed-window)
    - Has a memory of the context (thanks to the hidden layer's recurrent loop)
    - Same weights used for all inputs, so positionality is not overwritten (like in FFNN)
    > *Help capture more context while avoiding sparsity, storage, and compute issues* <br>
    > Done so by maintaing a hidden state that carries forward info from previous **time steps**, thus effectively capturing dependencies over time. <br>
    Unlike traditional models that rely on large, sparse feature representations, RNNs work with <ins>compact continuous embeddings</ins>, reducing storage needs <br>
    Its recurrent stucture allows it to <ins>share parameters across different time steps</ins>, which decreases computational complexity. <br>
    Thus, this design enables RNNs to model sequential data efficiently, capturing long-term dependencies without the overhead of storing extensive historical data or the need for large, sparse matrices.
    - **Time step**: the processing of a single word, incl. updates its hidden state, and moves to the next element. 
4. Issues
    - BPTT is slow to train
    - Due to infinite sequence, gradients can easily vanish or explode
    - Has trouble actually making use of long-range context
5. Hidden layer is the core of its contextual learning of words within sequences.
 > At each time step, the hidden layer updates its state based on the current input & previous hidden state, effectively summarising past info relevant to the current position in the sequence. This dynmamic updating allows the hidden layer to carry forward <ins>the semantic and syntactic context of words</ins>, thereby representing <ins>not just the individual word's meaning but its meaning in relation to the entire sequence</ins>.

## 3 LSTM Long Short-Term Memory
> A type of RNN that is designed to better handle **long-range dependencies** (长距离依赖性（long-range dependency）特指序列中远距离位置上的元素之间的依赖关系)
- Why proposed? 
    - Backpropagation through time (BPTT) process is expensive, instead of updating weights (adjusted to minise the loss by propagating errors backward through the network) after every time step (ie. every word), we do <ins>every T step</ins> (e.g. every sentence or paragraph). <br>
    - > Note that it's not equavalent to using only a T window size (n-gram) because we still have "infinite memory".
- The **vanishing gradient** in BPTT: the long path of multiplification makes gradient diminishes/increase exponentially, become either really small (then the far-away context will be "forgotten") or large (then the recency bias and no context - 指太关注最近的输入而忽略更圆的上下文)

## 3 Transformers

## 4 Prompting

## 5 Fine-tuning

## 5 RLHF

## 6 Agents

## 7 Probing Attribution

## 8 Behavioral assessment