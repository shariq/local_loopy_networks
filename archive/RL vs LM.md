# Tradeoffs

RL is better because:
- sample efficiency benchmark requires no code optimization; LM probably requires a lot of optimization
  + optimization is annoying but if you get anything at all working, Scott would be really excited if you demonstrate for equal # of updates local loopy networks perform way better - and he would love to collaborate on this
  + it would increase the timeline though
- closer to the end goal than supervised learning
  + if the goal is just to get the ball rolling, then forget about this: not clear if that's the goal or if the goal is to work towards AGI, or if the goal is to prove something to myself and others
- biological heuristics that this can work well on robotics tasks, vs no idea how fundamentally hard language modeling is
- SL is fine with offline learning, RL really wants as many flops as possible (via joschu) which it currently doesn't get. this would be a great way to give RL arbitrary # of flops
  + you could also do this by using an LSTM with adaptive computation time as your policy network: could be a cool project

LM is better because:
- LM is so much cleaner to setup and test and work with (i.e, shitty LM is a lot better than shitty RL)
  + counterpoint: just design good curriculum of RL tasks
  + countercounterpoint: a curriculum of RL tasks biases you towards certain kinds of problems; vs LM is way more agnostic
- RL has a higher dimensionality search space than LM (is reward global/local? is there a reward + error channel? if there's a reward channel how do we limit how much reward can be sent out by neurons to other neurons?) so it's harder to get something working
- SOTA deep RL seems to require hacks (value network + policy network, reward weighting, experience replay, exploration-exploitation hardcoding via epsilon schedule) which are harder to replicate under the single local loopy network modeling


# Solution

Do both, but one before the other.

Maybe something like:
- Do language modeling first; beat baseline as number of passes over data vs perplexity
- Offload code optimization and scaling up to Scott.
- Finish RL at the same time as Scott finishes code optimization.
- Publish both papers together.
