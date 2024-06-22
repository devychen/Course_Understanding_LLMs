## 1 Intro
- Begin with Machine Translation, now much broader: Name entity recognition, Tagging, Question Answering, Information Retrieval, Chatbots, etc...
![Image](/pics/history.png)

## 2 Pytorch
[SKIP]

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

### Gradient (of the loss)
> Vectors of partial derivatives of the loss function with respect to the model parameters. <br>
> - It suggests the rate and direction to update the parameters (in the direction opposite to the gradient, $<0$ increase, $>0$ decrease) in order to minimise the loss function. <br>
> - Matrix: $\frac{\partial L}{\partial V}$
- Define: specify the loss function first.
- Compute: compute the partial derivatives of the loss function with respect to each model parameter $\theta_j$​. The gradient is a vector of these partial derivatives.
- Update: use gradients to adject parameters - This is typically done using an optimization algorithm like Stochastic Gradient Descent (SGD).
- Iterate: repeat computing and updating for a number of iteration or until convergence

### Backpropagation
> A method (理解为*方法论*) used to compute gradients of a loss function. <br>
> - It leverages the chain rule of calculus to propagate the gradients backward through the network layers.

### Stochastic Gradient Descent (SGD)
> Optimization algorithm to minise the loss function by computing gradients iteratetively through mini-batches and update model parameters accordingly. <br>
> - vs. **vanilla/Batch GD** - single trial on entire dataset
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
> $LM$: A function that assigns to each input $X$ a probability distribution over $S$ given parameter $\theta \in \Theta$:<br> 
> i.e. $LM_A: X \mapsto \Delta(S)$ <br>
> - $X$: set of input conditions (images, texts in a different langauge) <br>
> - $S$: set of sequenes of tokens, also $W_{1:n} = \langle w_1, .., w_n \rangle$ <br>
> - $V$: vocabulary, set of tokens <br>
1. Causal LM (also left-to-right LM): 
    > a function that maps an initial token sequence to a next-token distribution: $LM: w_{1:n} \mapsto \Delta(V)$
    - next-token probability: $P_{LM}(w_{n+1}|w_{1:n})$
2. Training: 
    - minise next-word **Suprisal** - the uncertainty/unpredictbility of an event. Commonly measured by **Perplexity**
3. Prediction:
    - sample **Autoregressive generation**, using next-word probabilities
    > a method to generate sequences of data where each element is conditioned on previous elements. Left-to-right, not bidirectional.
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
        > - *Help capture more context while avoiding sparsity, storage, and compute issues* <br>
        > - Done so by maintaing a hidden state that carries forward info from previous **time steps**, thus effectively capturing dependencies over time. (**Time step**: the processing of a single word, incl. updates its hidden state, and moves to the next element.) <br>
        > - Unlike traditional models that rely on large, sparse feature representations, RNNs work with <ins>compact continuous embeddings</ins>, reducing storage needs <br>
        > - Its recurrent stucture allows it to <ins>share parameters across different time steps</ins>, which decreases computational complexity. <br>
        > - Thus, this design enables RNNs to model sequential data efficiently, capturing long-term dependencies without the overhead of storing extensive historical data or the need for large, sparse matrices.
4. Issues
    - BPTT is slow to train
    - Due to infinite sequence, gradients can easily vanish or explode
    - Has trouble actually making use of long-range context
5. Hidden layer is the core of its contextual learning of words within sequences.
    > At each time step, the hidden layer updates its state based on the current input & previous hidden state, effectively summarising past info relevant to the current position in the sequence. This dynmamic updating allows the hidden layer to carry forward <ins>the semantic and syntactic context of words</ins>, thereby representing <ins>not just the individual word's meaning but its meaning in relation 
    to the entire sequence</ins>.



## 3 Long Short-Term Memory (LSTM)
*[Short_Intro_Video](https://www.youtube.com/watch?v=YCzL96nL7j0&vl=en)* <br>
*[lecturer recoursed article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)*
> A type of RNN that is capable of learning **long-range dependencies** (长距离依赖性（long-range dependency）特指序列中远距离位置上的元素之间的依赖关系)
- Why proposed? 
    - Traditional RNNs struggle with long-term dependencies due to issues like **vanishing/exploding gradient**.
        > The **vanishing gradient** in BPTT: the long path of multiplification makes gradient diminishes/increase exponentially, become either really small (then the far-away context will be "forgotten") or large (then the recency bias and no context - 指太关注最近的输入而忽略更远的上下文). 
    - Instead of updating weights (adjusted to minise the loss by propagating errors backward through the network) after every time step (ie. every word), we do <ins>every T step</ins> (e.g. every sentence or paragraph). <br>
    - *Note that it's not equavalent to using only a T window size (n-gram) because we still have "infinite memory".*
- Intuition
    - RNNs run both LTM and STM through a single path, the **main idea** of LSTM units is that they solve this by providing separate paths:
    - Use two separate paths to make predictions: long-term & short-term
    - Use sigmoid & tanh activation function
        > - Sigmoid: turns any input into number between $0$ and $1$ <br>
        > - Tanh: into $-1$ and $1$
    ![image](/pics/afunction.png)
    - Integrate a memory cell that maintains information over time, and three gates (**input, forget, output gates**) that regulate the flow of information.
        - **input gate** controls what new info is stored in cell
        - **forget gate** determines what info is discarded
        - **output gate** decided what part of the cell state is used to compute the output (ie. the new STM from this enture LSTM unit)
        - **cell state** runs all the way across the top of the unit, represents the LTM
            -  the lack of weights allows LTM to flow through a series of unrolled units without causing the gradient to explore or vanish
        - **hidden state** represents the STM which are directly connected to weights that can modify them
        - we use Sigmoid to determine what % the LSTM remembers
        ![image](/pics/LSTM_3stages.png)
    - LSTM networks use the same weights and bias values across different time steps within the same layer. This **weight sharing** is a fundamental characteristic of recurrent neural networks, including LSTMs, and is what allows them to process sequences of variable length.
    - Strengths
        - Almost always outperforms vanilla RNNs
        - Capture long-range dependencies very well
    - Issues
        - More weights to learn, thus: 
        - Higher training data demand
        - Can still suffer form vanishing exploding gradients
    - Application
        - predicting next item (as above)
        - classification/regression: you can train the LSTM as a LM. By using the hidden layer that correspond to each item in you sequence, one can extract meaningful features that encapsulate the sequential dependencies and contextual info. These feature can then be fed into a classifier or regressor to perform task.

#### Bi-LSTM - *bidirection*
> LSTM that reads input bidirectionally, i.e. in both forward and backward directions. (Traditionally left-to-right)
    > - its final output concatenates the output of 2 separate LSTM networks at each time step
- Strength
    - usually performs at least = uni-directional RNNs/LSTMs
- Issue
    - Slower to train
    - Only possible if having full access to data

#### *Stacked LSTMs - *increasing abstractions*
> LSTM that consists of multiple LSTM layers stacked on top of each other. Each layer in the stack processess the output sequence from the layer below and passes its otput to the next layer above.
- Hidden layers provide an abstraction (hold "meaning"). Stacking hidden layers provides increased abstractions.
- **Increased abstraction** refers to the progressively more complex and high-level representations of the input data learned by each successive LSTM layer. The lower layers capture more basic, immediate, local dependencies, while higher layers can capture more global patterns and long-term dependencies.
- Better for complex sequence tasks.
![image](/pics/stack.png)

#### *ELM0 (Embeddings from Language Models)
> A deep contextualized word representation model developed by AllenNLP that captures complex characteristics of word use by training a bidirectional language model (BiLM) on a large corpus
General Idea:<br>
- Goal is to obtain highly rich embeddings for each word(unique type)
- Use both directions of context (bi-directional), with increasing abstractions (stacked)
- Linearly combine all abstract representations (hidden layers) and optimize w.r.t. a particular task (e.g., sentiment classification)
- Takeaway: given enough training data, having tons of explicit connections between your vectors is useful (the system can determine how to best use context)




## 3 Attention Architecture
*Advancement for handling sequential data.*
> A mechanism that allows models to selectively focus on different parts of input data when processing sequences.
    > - Why care: address limitations of LSTM (in captureing longer-range dependencies, early data could still get lost), by allowing models to dynamically focus on different parts of the input sequence.
    > - the **main idea** is to add a bunch of new paths from the Encoder to Decoder, one per input value, so that each step of the decoder can directly access input values.
1. Motivation: continuously think back at the originals while focusing on different parts.
2. *Some terms:
    - *Encoder: a NN that processes the inputs and encodes them into a set of continuous representation or context vectors. It decided which source parts are more important.
    - decoder: component that produces outputs
3. *At each decoder step, attention:
    - Recerived **attention input**: a decoder state $h_t$ and all encoder states $s_1, s_2, ..., s_m$;
    - Computes **attention scores**: <br>
    For each encoder state $s_k$, attention computes its "relevance" for this decoder state $h_t$. Formally, it applies an attention function which receives one decoder state and one encoder state and returns a scalar value $score(h_t, s_k)$;
    > Attention scoure doesn't have to be produced by FFNN, it can be any function.
    - Computes attention weights: a probability distribution - softmax applied to attention scores;
    - Computes attention output: the weighted sum of encoder states with attention weights
4. Summary of computation <br>
*Note that attention weights change from step to step
![image](/pics/attention_computation.png)
5. A higher view
![image](/pics/attention_higherview.png)
*[image resource](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#enc_dec_framework)*
6. Highlight
- It greatly improves seq2seq results
- Allows us to visualise the contribution each encoding word gave for each decoder's word

#### Self attention architecture
1. Motivation
    - Goal: to create great representations $z_i$ of the input; 
    - i.e. to weigh the importance of different words in the input sequence
2. Calculation steps:
    - Step 1: our self-attention head has just 3 $weights$ matrices $W_1, W_k, W_v$ in total, which are multiplied by each $x_1$ to create all vectors :
    $$
    q_i = w_q x_i \\
    k_i = w_k x_i \\
    v_i = w_v x_i \\
    $$
    - Under the hood, each $x_i$ has 3 small, associated vectors: $x_1$ has query $q_1$, key $k_1$, value $v_1$
    - Step 2: For word $x_1$, let's calculate the scores $s_1, s_2, s_3, s_4$, which represent how much attention to pay to each respective word $v_i$
    - Step 3: $s_1, s_2, s_3, s_4$ don't sum to 1. Let's divide by $\sqrt{len(k_i)}$ and softmax it and get the $a_i$
    - Step 4: Weight $v_i$ vectors and simply sum them up:
    $
    z_2 = a_1 * v_1 + a_2 * v_2 + a_3 * v_3 + a_4 * v_4   
    $
    - Strength
        - Context-aware
        - Much more powerful than static embeddings
    - Issues
        - Slower to train
        - Cannot easitly be pre-computed


## 3 Transformers Architecture
[an illustrated text tutorial](https://jalammar.github.io/illustrated-transformer/) <br>
[Attention is all you need](https://arxiv.org/pdf/1706.03762) <br>
*A paradigm shift from recurrent structures by entirely relying on self-attention mechanisms for both encoding and decoding*
> A NN design that processes sequential data using self-attention mechanisms instead of recurrent layers.
    > - Ultimately, just to produce a very contextualised embedding $r_i$ of each word $x_i$.
1. Steps
    - Additional steps: further pass each $z_i$ through a FFNN
    - **residual connection**: is concated to help ensure relevant info is getting forward passed.
    - **LayerNorm**: performed to stabilised the network and allow for preper gradient flow, also should do after FFNN
    - Each $z_i$ can be computed **in parallel**, unlike LSTM
    - $r_i$ are now the new representations, and this entire component is called a **Transformer Encoder** (- processes input sequences to produce context-aware representations)
    - Multi-headed self-attention, each head produces a $z_1$ vector and then concat.
    - Stacked encoders, not only one.
    ![image](/pics/transformer.png)
2. Encoders
    - Encoders produce contextrualised embeddings for each word.
    - Decoders generate new sequences of text.
    - Decoders are identical to encoders, except that they have an addtional attention head in betweent the self-attention and FFNN layers.
        - This additional attention head focueses on parts of the encoder's representations
    ![image](/pics/transformer_decoder.png)
    ![image](/pics/transformer_en_de.png)
    ![image](/pics/transformer_model.png)
3. Complexity
    - Compare self-attention, recurrent, convolutional, self-attention (restriced)
    - Note: when learning dependencies b/w words, you don't want long paths. Shorter is better.

#### BERT
> Bidirectional Encoder Representations from Transformers
1. Motivation: bi-directional 
2. Training objectives:
    - Predict the masked word
    - Two sentences are fed in at a time. Predict if the 2nd truly follows the 1st.
3. Architecture
4. Strength
    - Great for embedding learning
    - Powerful for transfer learning to other tasks
5. Issue
    - Not easily generative (no decoders)

## 4 Sequence Model
> Model that processes and/or generates a token sequence
    > - RNN, LSTM, Transformer...
- Autoregressive/left-to-right models vs Bidrectional
    - Autoregressive/left-to-right: 
        - Process from start to end, predicts/generates based on past elements only. 
        - Example: RNN, GPT
    - Bidirectional
        - Process the entire sequence in both forward and backward directions simultaneously, capture context from both past and future elements.
        - Example: LSTM, BERT
- One-to-one, one-to-many, many-to-one, many-to-many
    - Maps a single/sequence of input(s) to a single/sequence of output(s)
    - Autoregressive & masked are many-to-many.

## 4 State-of-art LLMs
- Kinds of LLMs
    - Core LLMs (foundation models): predict statistically likely next token
        - Example: GPT2, LLaMA2/3 (LLaMA for Large Language Model Meta AI)
    - Prepped LLMs (assistants): fine-tuned (e.g. RLHF), predicted token likely to please the user
        - Example: GPT3.5, LLaMA3 Instruct
    - LLM-based applications (agents): algorithm using LLMs
        - Example: ChatGPT
- [SKIP] A short intro to building LLaMa, how to train it


## 4 In-context learning (ICL) 
*aka. k-short learning*
> A model's ability to understand and perform tasks based on the context of input, without requiring explicit training for each specific task.
- Propmting with $k$ pairs of demonstrations (input $x_i$, and the corresponding output/label $y_i$). When ready to test, simply add only the target input $x_t$ of test item ($x_t, y_t$) to the prompt.
- Boost on common tasks, performs well <ins>w/o task instructions</ins>
- The performance tends to increase with the number of demonstration pairs $k$ provided. (ie. the more examples, the better)
- What matters for it to work: 
    1. Structure of simple prompt (so ICL need not be genuine learning)
        > **"Genuine learning"** refers to the traditional process where a model learns from extensive labeled training data. VS. in-context learning, where the emphasis is on the model's ability to perform tasks effectively based on minimal examples.
    2. Similarity of demonstrations we provided and target outcomes we expect
        - **(When these are closely aligned, the model can leverage statistical cues effectively, enhancing its performance)*
    - Good explanations also help
    - [NOT] ~~instructions and label correctness~~
- k-short examples may provide statistical cues which the LM can exploit.
    - **(ICL isn't necessarily reliant on explicit instructions or perfectly labeled data—it's more about the structure and relevance of the input provided to the model)*

## 4 Prompting Engineering
> the strategic design and optimization of inputs given to AI models to guide their behavior and improve task performance.

#### Chain-of-Thought prompting *CoT*
> Guiding our language models through a series of prompts or questions that build upon each other logically
- Start by: giving clear task instructions (to set the stage)
- Then: give ≥ 1 examples with explicit chain-of-thought reasoning leading to the correct answer
    - *说人话: each step in the chain of prompts leads the model towards the correct answer through reasoned steps*
- Particularly effective for complex tasks, vs. few-shot prompting alone. 
    - However, it's crucial that we do "right" task analysis (to ensure that each step in the chain of thought is guiding the model in the right direction.)


#### Zero-shot prompting w/o CoT: 
- May incl. task instruction, but no example/illustation beforehead
    - zero-shot chain-of-thought (CoT) prompting extends this concept by guiding the model through a sequence of prompts or questions that logically lead it towards the correct answer
- works well (when no prior training examples): models find-tuned on instruction-following data; frequent (simple) tasks

#### Few-shot prompting w/o CoT
- Provide a small number of examples or demonstrations before tasking the model to perform a task. The model learns from these examples to generalize and generate responses for similar tasks.
- Particularly useful in situations where there are some examples available but not enough to fully train the model.

#### Emsemble methods
> Involes combining multiple machine learning models to improve overall performance and robustness
1. **Self-consistenct**
    - *Ensuring that the individual models within the ensemble are consistent with each other in their predictions or behaviors
    - e.g. few-shot CoT + self-generated CoT sequences (greedily)
2. **Generated knowledge prompting**
    - *A technique where ensemble methods prompt models to generate new knowledge or insights by aggregating outputs from diverse models within the ensemble


#### Task-compostion/neuro-symbolic generation
> Involves combining neural networks with symbolic reasoning techniques to generate outputs
- Tree of thoughts
    > A method of organizing and representing information hierarchically, resembling a tree structure where each node represents a concept or idea, and branches depict relationships or connections between these concepts.


## 5 Fine-tuning

## 5 RLHF

## 6 Agents

## 7 Probing Attribution

## 8 Behavioral assessment

