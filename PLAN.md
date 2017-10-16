# Plan of attack

step 0 is finding a loopy learning rule which even works on super simple
problems.

step 1 is beating a baseline on some machine learning problem.

step 2 is getting SOTA on something which is a big deal. this is the acceptance
criteria for this approach.

step 3 is publishing the results.

for a language model, how input/output happens:
because we want the loopy network to compute for arbitrary amounts of time,
the output nodes will be a one hot encoding of all characters + a node to
determine when to read the output (read the output on this node switching from
low to high)
during training, when the network indicates it has output a character, we feed
another character in on the next time step, for one time step, and the output
nodes get an error signal
the input to the network is also one hot encoded

all loopy networks must support being initialized with:
- arbitrary number of hidden state nodes
- arbitrary number of input nodes
- arbitrary number of output nodes

some ideas for parameters:
- connectivity rule (how the graph gets connected initially)
- degree of connectivity
- learning rate
- the learning rule, and its parameters (number of channels per edge, ...)

then, the super interesting bit will be to come up with a bunch of tests.
these tests will be used to sweep the parameter space.
they're pretty fast to run but test different kinds of things.
some ideas for tests:
- just memorizing longer and longer random sequences
- make a small random circuit which generates a sequence
- explicitly test for long term learning - e.g, 0,0,0,....,1
- robustness of the network to being fed garbage, in different forms
- robustness of the network to randomly perturbed weights
- robustness of the network to randomly removed edges/nodes
- learning works with multiple size networks
- no catastrophic forgetting - so train on X then Y, then test X
- transfer learning - learn circuit X, see if you can fine tune to X + delta
- compositional learning - learn A, B, C, ... Z circuits, then learn composition
- self learning: learn X, then try to learn a faster approximation for X (RL focused)
- learn a function which has a lot of redundant circuit components, like a convolution

we don't expect all hyper parameters to try and be scale invariant
so we will fine-tune hyper parameters as we increase scale of network + compute



# Technical details

must have a lot of basic tests!!!!
because this code will be notoriously easy to have loads of logic bugs in
would be nice to have visualizations, not necessary


### step 0
##### prove it even exists
lots of experiments, lots of tests, simple validation, etc.
find a local loopy network which learns at all - big deal
easy to modify loopy network library
may or may not need to do a huge search over rules
write tests to ensure rules behave as expected!!!!!!!!!!!!!!!!
(especially before doing a *conclusive* big search)


### step 1
##### beat a baseline
this will involve further finetuning, but hopefully not starting from scratch.
may require better connectivity graphs and so on.
probably implement all the crazier tests here (transfer/compositionality/...)


### step 2
##### get a SOTA result
talk w/ Scott Gray
maybe implement custom optimizer on pytorch
