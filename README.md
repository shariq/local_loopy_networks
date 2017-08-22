# Large Loopy Networks
###### goal: beat some significant state of the art metric in 4 months (probably language modeling) using a large "loopy" network

## Overview

modern recurrent models tend to have a single "loop" of learning/recursion
hypothesis: this is not "good" for representing certain kinds of functions
e.g, long/medium term memory
and empirically not "good" at learning certain kinds of functions
(but we don't have a good theory of learnability)

instead if we have a model with many "loops", which all touch each other...
so kind of like a big randomly connected directed graph, with a lot of cycles
let's call this a loopy network
then maybe this is "good" at representing long/medium term memory
and the crazy idea:
maybe there exists a local learning rule to learn these kinds of functions
the rule being local is biologically and physically inspired

with a computational prior this would be formalized approximately as
a random LSTM is further away from random code than a random loopy network

with a physical prior this would be formalized approximately as
a random LSTM is further away from random physical function than loopy network

anyways this formalization has a lot of holes, so let's ignore it for now;
more importantly is to get this working, then go back and try to find out why


## Higher purpose

i think some kind of many-loop local learning rule model is going to be the
last class of model before computers get way better at thinking about thinking
than us. and i think once we have good many-loop local learning models, we can
start setting up really interesting problems like automated theorem proving as
a game played between agents without having to hardcode almost any of the agent.

and then from there we can start working towards AI which does science and
engineering, and build the most empowering/valuable/destructive technology
available to humanity! hopefully in a way which results in humanity prospering,
the end of disease, the end of death, exploration of the galaxy(s), and a large
amount of meaning and regret minimization for all individuals


## Plan of attack

eventually i think something like this would make a lot of sense in RL,
unsupervised learning, supervised learning, and generally arbitrary learning
problems; where it'll be just a single model we use for all of these

but to start i'm tentatively interested in applying this to language modeling,
and other sequence prediction tasks.

so my goal is to have a state of the art language model within 4 months.

i think scott will be interested in this problem (biologically plausible
learning) and i'll try to collab with him to run on CUDA. probably will take a
month.

how input/output happens:
because we want the loopy network to compute for arbitrary amounts of time,
the output nodes will be a one hot encoding of all characters + a node to
determine when to read the output (read the output on this node switching from
low to high)
during training, when the network indicates it has output a character, we feed
another character in on the next time step, for one time step, and the output
nodes get an error signal
the input to the network is also one hot encoded

i want to parametrize the model class of loopy networks for seq2seq.

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
- no catastrophic forgetting - so train on X then Y, then test X
- transfer learning - learn circuit X, see if you can fine tune to X + delta
- compositional learning - learn A, B, C, ... Z circuits, then learn composition

we don't expect all hyper parameters to try and be scale invariant
so we will fine-tune hyper parameters as we increase scale of network + compute


## Technical details

initially we are just trying to get this to work, and don't care about running
fast or with low memory footprint. so it'll all be written in nice clean python.

i'll make a framework in python for loopy networks and attaching learning rules
to them, and attaching input/output, and running train + test

another set of code to run a lot of different tests on these models, for
filtering on learning rules


## Author

the current sole contributor to this work is Shariq Hashme:

https://shar.iq/
