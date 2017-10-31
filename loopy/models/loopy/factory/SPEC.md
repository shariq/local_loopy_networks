REMINDER: you can reuse parts of the ExpressionTree in itself to make the distribution of complex ExpressionTrees nicer (i.e, if we use node_memory[0] > edge_memory[0] * 2 then it should be somewhat likely we use it multiple times and not just once)

number of signal channels
(automatically propagated)

additional edge memory and node memory is made based on the primitives and functions used in the update/initialize rule. the generated memory is all placed near the beginning; the rest of the memory is randomly sized and we remember the offset for that so we can easily evaluate the tree

trees

update rule + initialize rule primitives and functions:
- multiply(a, b)
- add(a, b)
- subtract(a, b)
- abs(a)
- primitives for operating on vectors:
    - reduce operations; forced when sticking into single variable
    - elementwise operations with automatic resizing; requires same type of vector (finite number of types, defined up front)
    - filter(filter_number, signal | error | edge_memory_index | filter_expression) -> for that filter number, gets the corresponding edge values (filter_expression is the filter_expression used to make that filter)
- >=, <=, ==
- uniform_random(-0.5, 0.5)
- gaussian(0, 1)
- constants: -1, 0, 1; 0.1, 0.01, 0.001, -2, 2
- clip(a, -1, 1)
- relu(a)
- negative(a)
- sigmoid(a)
- tanh(a)
- mean(a)
- stdev(a)
- sin(a)
- cos(a)
- sgn(a)
- node(a) (node memory at some a; a is picked randomly when generating node(a))
- edge(a) (edge memory at some a for all edges; a is picked randomly when generating edge(a))

- functions which are error prone; so don't include them:
    - exp(a)
    - log(a)
    - sqrt(a)
    - divide(a, b)

we have all these operations; and we need to assign the values of the operations to channels and memory

also our tests will contain the code for network size and topology and that's randomly made up; also contains code for generating dataset; but spec will contain code for train, backward, forward (randomly selected from some set of these which we have)

tests initially needs to be just subtract + add
