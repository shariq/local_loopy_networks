# October 24, 2017

Oct 16 - 24 has had little to no useful work. I had been trying to think of a good way to parametrize the search space. Time to make a decision... I think what needs to be done is I just need to use the best thing I thought of, in some interesting learning setting(s). Learning setting(s) defined as input/output/topology rules. Initially I want to find a learning rule which works with some sampled Waxman graphs (located in loopy.py); and input/output are sent in one/more obvious ways; and the whole thing is forced to use some signals and error. It would be nice to do everything with one signal, but forget that crazy idea for now: we can easily modify the tests once they exist to try this and other ideas. Much more important to get something kind of OK working, then to make it better through new knowledge gained. Bad pattern I find myself consistently falling into - planning too much instead of doing enough - thankfully I'm not kind of aware of it and can try to consciously prevent it.

I want to have a working learning rule by Thursday (Oct 27): so I can share at the OpenAI alum dinner! So today I finish a shitty search implementation; tomorrow I run it a bit, make obvious improvements, make it distributed; run overnight on 200 servers and cross fingers!!!

I will implement the rules as some kind of formal grammar. One of the rules will be to assign a value to a variable. If a variable is used before it has something assigned to it, it will be automatically assigned something from the rest of the graph. This is how reuse of expressions will occur.

So that's pretty easy and will easily give us a nice string describing our code. Maybe we'll modify to have the rules all have some weight, but past that all rules will be context free and probability will be context free (otherwise we will drown in complexity).

Once we have a nice string, we need to turn that into some code. To make life easier we can probably just have the string be the code; if that doesn't work for some reason we can do a simple post processing step. Post processing step would preferably have two passes, one to gather some data about the rule, another to do a string replace. (e.g, memory sizes)

We will also need a nice test harness, which we currently don't have. It must support a input and output forms and a few different graph topologies, preferably of different sizes. The tests will be simply training AND, OR, and adding and subtracting (two different subtractions). We will need a way to limit how long the test runs for before it's declared a failure: this logic goes in the input and output code, and raises a loopylimit exception.

Once this is done, we test by generating a bunch of rules and seeing what the output looks like. We maybe add a few small changes to make the outputs look nicer more often, like limiting string length or making reuse more likely, or changing how vector operations work like throwing away rules which use nonmatching vector lengths (mainly filtering edge memories/using signals), changing probabilities on rules being applied.

Then it's time to sleep.

# October 16, 2017

Successfully implemented and tested backprop as a local learning rule, and wrote a nice loopy module.

Starting to see some unexpected annoyances, but this project is still very exciting.

These unexpected annoyances are in the context of: next we want to write some large number of update/initialization rule agnostic tests, and have them work on some reasonable percent of learning rules. We also want to parametrize over a large number of update/initialization rules, and have that parametrization find interesting rules with some amount of regularity: for instance, rules with only a single channel for both error and signal, and unsupervised rules which find structure in data and can be sampled.

## Unexpected annoyances

### Arbitrariness of picking specific channels

So all the harness code makes a lot of assumptions about how the channels are laid out. Specifically to send errors/inputs in it has to know about has_signal, has_error, signal, error; and for signals it has to do that on the nodes too! This is really crazy.

Unfortunately there is nothing obvious I can think of that we can do about this, and the outer test code has to probably work with the model only if the channels are laid out in a specific way. Unless... we put a linear feedforward layer in between! Just kidding :p The real way around this is to start by testing all learning rules with really really simple problems, and then move up from there. And to evolve good learning rules into better ones instead of always randomly generating them.

### Large amount of harness code

There is a lot of harness code around the model which was required to get the simple feedforward 1 layer leaky ReLU backprop to work.

Specifically:
- beginning a backward pass requires sending error along all the edges of the output, which involves looking at the gradient and setting both error + has_error channels on corresponding edges
- beginning a forward pass requires setting signal + has_signal channels on corresponding edges and nodes! it also involves checking that all output nodes have a signal available to see when to stop stepping and sample the output nodes. it also involves grabbing the signal from all the output nodes by knowing which channel has the output...
- there is a reset step which gets rid of a lot of cruft between training runs. we should get rid of this by better managing state of error + signal has/has not on nodes when they get a backwards pass
- we have a very specific graph topology to make backprop work: first, distance from input-output and backwards is equal along all paths which can be traversed; second, there is a "secondary output layer" to make it easier to send a gradient back; third, there are no loops
- the learning rate alone ended up making a huge difference. so perhaps instead of checking whether the entire test of training on a bool function is passed, we can also check progress along the way.

### Very long update and initialize rule

Also, the update rule and initialize rule have a LOT of code! Probably we can come up with some nicer primitives to reduce the amount of code, like a signal channel which comes with a "has_signal" channel, or the mean/stdev of a channel, or operations on a large number of edges at once. We can probably also reduce the amount of code in these two methods just by rewriting with this in mind. While rewriting we can look for nice abstractions for similar logic.

## Revised plan

Get rid of clean(): generally any kind of global reset is antithetic to the idea of a long running learning process.

Two directions:
- Make learning work better by objective metrics and with new tests:
    - extend backprop to work on sequence learning; so new ways of sampling/backward pass (specifically we mean language modeling; not seq2seq)
    - implement tests like mnist, shifting distribution learning, compositionality, learning from small random circuit and growing that circuit, learning explicit long term dependencies, robustness to garbage data, robustness to randomly perturbed/zeroed out edges, learning simple functions with very large models quickly, function with a lot of redundant components like a convolution
    - nice abstraction over update and initialization rules
    - everything necessary for a big search
- Get learning to work under more physically plausible/beautiful constraints for input/output/error/topology combinations.
    - make backprop more modular so we can replace input/output/error/topology with different ones (e.g, unsupervised learning with no error; input and error sent on same channels; etc)
    - implement many different kinds of input/output/error/topology rules; and ways of combining them better (e.g, send input and expected output at the same time; so split up sending from stepping)
    - implement progressively more difficult tests but not testing new properties, just whether model can actually learn on more difficult problems - so like mnist
    - big search

For nice abstraction over update and initialization rules - seed ideas: tree edit distance, optimization, constraints, physics inspired, high dimensional functions, ...


# October 1, 2017

Beginning of project. Brain dump of how I think things could go.

## Plan of attack

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


## Technical details

must have a lot of basic tests!!!!
because this code will be notoriously easy to have loads of logic bugs in
would be nice to have visualizations, not necessary


#### step 0
###### prove it even exists
lots of experiments, lots of tests, simple validation, etc.
find a local loopy network which learns at all - big deal
easy to modify loopy network library
may or may not need to do a huge search over rules
write tests to ensure rules behave as expected!!!!!!!!!!!!!!!!
(especially before doing a *conclusive* big search)


#### step 1
###### beat a baseline
this will involve further finetuning, but hopefully not starting from scratch.
may require better connectivity graphs and so on.
probably implement all the crazier tests here (transfer/compositionality/...)


#### step 2
###### get a SOTA result
optimize, maybe implement on GPU
