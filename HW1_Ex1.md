Please answer the following exercises. Importantly, please reason step by step; i.e., where calculations are required, please provide intermediate steps of how you arrived at your solution. You do not need to write any code, just mathematical solutions._

[6pts] Consider the corpus $C$ with the following sentences: $C=$
{“The cat sleeps”, “The mouse sings”, “The cat sleeps”, “A dog sings”}.
1. Define the vocabulary $V$ of this corpus (assuming by-word tokenization). 
2. Pick one of the four sentences in $C$. Formulate the probability of that sentence in the form of the chain rule. 
Calculate the probability of each termn in the chain rule, given the corpus.

[4pts] We want to train a neural network that takes as input two numbers $x1, x2$
, passes them through three hidden linear layers, each with 13 neurons, each followed by the ReLU activation function, and outputs three numbers $y1, y2, y3$
. Write down all weight matrices of this network with their dimensions. 
(Example: if one weight matrix has the dimensions $3\times5$, write $M_1\ \in R^{3\times5}$
)

[2pts] Consider the sequence: “Input: Some students trained each language model”. Assuming that each word+space/punctuation corresponds to one token, consider the following token probabilities of this sequence under some trained language model:
. Compute the average surprisal of this sequence under that language model. (Note: in this class we always assume the base e
for log, unless indicated otherwise. This is also usually the case throughout NLP.)