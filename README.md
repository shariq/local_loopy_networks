# Large Loopy Networks
###### goal: beat some significant state of the art metric in 4 months using a large "loopy" network

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


## Author

Shariq Hashme
