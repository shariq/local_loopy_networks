# November 18, 2017

Started working at Scale, so didn't have much time recently. But all the bugs are fixed and now we need to improve defaults, and make the code faster. Today I'll get this running on some droplets.

Got code running on some droplets, dumping results to postgres. Potential next steps:
- shared vs nonshared edge memory
- edge parity
- default filters and conditionals
- get backprop working (so we test that rule rendering is behaving sanely; try to hit rule rendering code paths here)
- write complexity decreaser and evolver for use with backprop
- write better tests, which resolve faster
- include more things in pg log - train time, scores for all tests.
- improve performance (not sure why running such a simple model is so slow right now)
- scale up a bit: make sure to first make this cost effective by benchmarking across a few different cloud providers, and checking for D.O. which droplets are most cost efficient.
- long term: dashboard to monitor performance, pool of models to evolve, interactive evolution with dashboard that lets you modify hyperparams and fork (keeps track of full tree of changes), more reuse in expression trees, ...
- crazy ideas: symmetric error/signal, reinforcement learning, reimplement on GPU, somehow get access to 10k+ cores, make code run asynchronously on a GPU, reimplement slow parts in C/C++,
- to get RL/weird SL structure to work, we need to try and have a similarly working curriculum for supervised learning from scratch, without starting from backprop.

# November 8, 2017

I'm starting to be more OK with just working a lot at a consistent pace without being super productive. Maybe the path to being super productive is accept where I'm at now, and push incrementally to be more focused.

Today I got the code generation running and compiling. Now there's a few bugs which need to be fixed but we are basically done with generating and testing update rules. Hopefully we finally run the experiment tomorrow.

There is a lot of room for improvement; this is only the beginning of the project.

# November 7, 2017

I went to NOLA for a wedding and thought I'd finish this there. Unfortunately I did not spend anytime at all on actually working, only thinking. Right now all the harness code is significantly biased towards backprop-like learning - how can we make this more general?

In general I'm very pessimistic that this in its current form will find anything extremely interesting. I think there needs to be better defaults around how signals are propagated, and better ways of putting error signals into the network (e.g, don't assume a forward/backward phase - there needs to be a way to do them both at the same time - maybe there's a bunch of other crazy ways of doing this like applying error all over the place directly without it being propagated by the circuit itself - idk just lots of ways in which this is fragile to our assumptions about how a big circuit should learn).

# November 2, 2017

The rules are actually looking really interesting! :-) After looking at one or two random ones, I was curious what the sin of a gaussian variable was since that was one of the initializations! And it seems like when rules do something dumb it's not so dumb that it'll break stuff - which is wonderful :-) mostly it just means that part of the expression is ignored or useless. Probably most of the actively harmful stuff is not easy to see by eye, and just needs to be run.

Finished rule generation logic, still need to add some default conditionals and default filters; after need to write rendering logic and test. Most rendering is easy; filter select and blow up may be hard; filtering will be a little messy (just keep track, for each filter, of a bunch of indices corresponding to what that filter evaluated to near the beginning); conditionals will be a bit tricky but not that much (just > 0 is fine for now; if it doesn't work it just means our rules are less complicated); rest is EZ :-)

Will try to finally set this up on 200 machines tomorrow. Looks like pickled models are only 40-60kb, so hopefully can stream any pickled model which can pass the first test over; while also having some way of tracking number of jobs completed by each individual worker. Fingers crossed!!!

After that there's a few adjustments I could make:
- edges have read + write memory; writes to read memory just get ignored (lol); write consists of node-only + edge-shared; read consists of other node's edge memory on this shared edge. whole thing gets treated the same by rules, except the edge_memory_size number is different. the synchronization of edges would be different. we would definitely want to make sure that the edge memory gets seen in normal/flipped mode by nodes on either side of the edge: this way rules behave the same on both sides.
- edges have parity - parity just gives you a default way to align signals. input flows from low number nodes to high number nodes; output comes out at the highest number nodes. so parity would just tell you: -1 if other node on this edge is smaller, +1 if node on this edge is bigger. Make sure to make a default filter for this, maybe in both directions.
- write the decrease_complexity method. maybe by just having it turn a random operator into a leaf. still a lot of annoying filter/type/base expression/etc weirdness. then you can write an evolutionary thing to search over rules which have worked.
- implement backprop as a ruleset - :'( . but this would potentially allow us to fix the backprop bug with some kind of evolutionary approach on the existing ruleset
- write some additional tests to more quickly test if a randomly generated rule is good or bad. for instance, check that the output of a 1 input 1 output network is affected by the input in a sensible way.


# November 1, 2017

Did not hit goal again, need to sleep early to wake up early.

Pretty close to having something, only need to implement filter/slot_type/expression_type/base_expression/is_reducer/number_children logic/operator_input_type (operator_input_type hasn't been defined yet but probably should be; we should let the operator decide what kind of input it wants) and also all the code for training, forward pass, backward pass, etc - this should be done in a regular class which is then subclassed by Model.

The way the type stuff will work is... we come up with a list of constraints.
- reducers have to take in vectors
- generally we want to take in vectors more than floats, maybe 2:1
- two child op can have one child randomly be float and other be vector
- make sure leaves which require a vector end up going to an op which expects a vector (op decides before making the leaf)
- if we are the root and assigning to a vector, we should definitely output a vector and not a float (meaning not something which is actually a float). except for initialize expressions, then outputting a float is fine.
- if we are the root and this is a conditional expression, it needs to output a float. the float is tested for being above 0 as "true". this is literally crazy because of how big it makes the search space

The way the filter stuff will work is...
- if the output type of an expression is a float, it doesn't have a filter
- filter is inherited from parent operation
- if filter is None and expression returns a vector, then there's some probability with which filter gets selected
- if filter has a value and expression returns a vector, it has to stay the same in all children, except float children

The way the base_expression will work is...
- some P(base_expression), sampled from [0.25, 0.5] * 10, [0.0, 0.25] * 1 for each rule
- ends up being used in a leaf with this P; base expression is not for an operator (filter is defined differently and not part of base_expression - it must be used because of slot_filter of root)

Note: always support logic being run on root of tree which has parent=None

After implementing this, generate a large number of rules and find dumb stuff. Make sure to consider the full context in which it is dumb (expression_type, root/child, tree_depth, slot_type, filter, is_reducer, number_children, operator_input_type, ...)

Then of course there's the question of actually running this on 200 machines... Would be nice to just have a DB which supports 200 incoming connections of 10MB/s, but not sure how realistic that is (ethernet capacity).


# October 30, 2017

October 24 - October 30 involved figuring out the implementation details of generating rules. Turns out formal grammars are good for describing parsers but not for describing how to parametrize over a large number of strings. What I thought would be working by Oct 26/27 is still not working... I think if I didn't get distracted by Halloween parties/hanging out with people I would've been done by now. Something to work on long term: working longer on mentally draining work. I do it when I like it and I'm not distracted. Distractions today are a different class than those of the past.

Anyways I'm pretty close to setting up the first experiment now... Some TODO items are to rename the pip module to local_networks; since the module mostly deals with local updates, and then have a submodule be called loopy where most of the experiments into loopy local update rules happen.

I did write the tests and surrounding testing code which is nice.

Figuring out how to implement filters in a nice way, and thinking of conditionals as a nice way to make update rule look like our backprop model, was pretty important. And this figuring out only happened because I started implementing things and thinking using the universe computer. Need to do this more frequently and sooner in the future.

Still need to hook up operators/filters/conditionals in expression tree generation, enforcing type constraints, enforcing constraints based on expression_type, rendering trees correctly with context for leaves, reusing parts of the ExpressionTree, and using sane probabilities for different groups of expressions. And probably some other things I'm not thinking of.

Once the rules are being generated properly, there's some additional scaffolding which needs to be written in the Model class. The regular backprop stuff but also any scaffolding around how signals are propagated around (not sure yet what this is: maybe just keeping track of sent_signal and has_signal - also we can ignore zero signals with some probability to prevent catastrophic failure).

Then all that stuff needs to be tested to see if it behaves as expected. Some tests could be: trying to implement backprop as a Ruleset, testing if trees generated have the right complexity (right now before implementing increase_complexity correctly they don't, and it's especially important at low complexities where all filters seem to be only one node big), running a randomly generated tree and stepping through the edge/node/signal of a few different nodes to see if they behave as expected including at input/output times. Need to think of better tests than this.

While doing this there will be a lot of tinkering to adjust past ideas to get better rules.

Then we run a big search on Digital Ocean. Need a way to aggregate data; maybe a big DB which can handle O(200) connections. Test with a few nodes before - package in Docker, run on droplet with droplet API. ~$17 gives us 30s * 1,000,000, which means O(1M) - O(10M) rules tested for $17. So maybe spend $17-$50 on initial run. If that fails, run the same test on forward topology graphs with the same randomly generated distribution; if that fails, run the test on loopy topology graphs but with permutations of backprop. If all 3 fail, this is the end of the road. Else, the beginning of a long road of improving those rules, evolving them, applying to RL/LM, optimizing code, etc.

One weird thing about backprop model is nested conditionals. Hopefully this isn't too much of an issue. We could also nest conditionals in our ExpressionTree? Ew - only do this after trying to implement backprop as an ExpressionTree and failing.


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
