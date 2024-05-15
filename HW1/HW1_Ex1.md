**Please answer the following exercises. Importantly, please reason step by step; i.e., where calculations are required, please provide intermediate steps of how you arrived at your solution. You do not need to write any code, just mathematical solutions.**
## Exercise 1
### 1
**Question**[6pts] <br>
Consider the corpus $C$ with the following sentences: $C=$
{“The cat sleeps”, “The mouse sings”, “The cat sleeps”, “A dog sings”}. <br>
(a) Define the vocabulary $V$ of this corpus (assuming by-word tokenization). <br> 
(b) Pick one of the four sentences in $C$. Formulate the probability of that sentence in the form of the chain rule. 
Calculate the probability of each term in the chain rule, given the corpus.

**Answer** <br>
(a) <br>
```{math} 
V={"The", "cat", "sleeps", "mouse", "sings", "dog"} 
```
(b) <br>
$ S = {"The\ cat\ sleeps"} $ <br>
$ P(The) = \frac{3}{12} = 0.25$ <br>
$ P(cat) = \frac{3}{12} = 0.25$ <br>
$ P(sleeps) \frac{2}{12} = \frac{1}{6} \approx = 0.17 $ <br>
$ P("The\ cat\ sleeps") = P(The)P(cat)P(sleep) \approx 0.01  $ <br>

### 2
**Question**[4pts]<br>
 We want to train a neural network that takes as input two numbers $x1, x2$, passes them through three hidden linear layers, each with 13 neurons, each followed by the ReLU activation function, and outputs three numbers $y1, y2, y3$
. Write down all weight matrices of this network with their dimensions. 
(Example: if one weight matrix has the dimensions $3\times5$, write $M_1\in R^{3\times5}$) <br>

**Answer** <br>
Given:<br>
- 1 input layer, 2 inputs/neurons $x1, x2$ <br>
- 3 hidden layers, 13 neurons each <br>
- 1 output layer, 3 neurons $y1, y2, y3$ <br>
From input layer to 1st hidden layer: $M_1 \in R^{2\times13}$ <br>
From 1st hidden layer to 2nd hidden layer: $M_2 \in R^{13\times13}$ <br>
From 2nd hidden layer to 3rd hidden layer: $M_3 \in R^{13\times13} $<br>
From 3rd hidden layer to output layer: $M_{out} \in R^{13\times3}$ <br>

### 3
**Question**[2pts]<br>
 Consider the sequence: <br>
 “Input: Some students trained each language model”. <br> 
 Assuming that each word+space/punctuation corresponds to one token, consider the following token probabilities of this sequence under some trained language model: $p=[0.67, 0.91, 0.83, 0.40, 0.29, 0.58, 0.75]$. Compute the average surprisal of this sequence under that language model. (Note: in this class we always assume the base $e$ for $log$, unless indicated otherwise. This is also usually the case throughout NLP.)

**Answer** <br>
Calculate the suprisal for each token and then compute the average.
Given $p=[0.67, 0.91, 0.83, 0.40, 0.29, 0.58, 0.75], we have the suprisal for each token with log base $e$: <br>
$Surprisal(p)=[−ln(0.67),−ln(0.91),−ln(0.83),−ln(0.40),−ln(0.29),−ln(0.58),−ln(0.75)]$ <br>
And the average: $Surprisal(p)\approx [0.400,0.093,0.186,0.916,1.237,0.544,0.288] \approx 0.523$ <br>
