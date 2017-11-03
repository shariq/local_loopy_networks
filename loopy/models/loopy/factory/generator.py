import random
from collections import defaultdict
import numpy as np

import loopy.models.loopy.factory.leaves as leaves
import loopy.models.loopy.factory.operators as operators

# the way we make an expression of some complexity is to take a blank expression and randomly distort it, and keep doing that until we get our desired length
# increase_complexity should sample from the right distribution; so adding one node which may increase complexity by too much at once; but it's all approximate anyways so it's fine if it's a bit off

# operators, slots, and other things like that (filtering) are defined somehow outside the scope of this code
# code generated here should try as much as possible to only be the parts which are unique between different runs of generator.py, except when that reduces readability

class Harness:
    def __init__(self, ruleset_generator=None):
        if ruleset_generator is None:
            ruleset_generator = Ruleset
        self.ruleset_generator = ruleset_generator

    def sample_node_memory_size(self):
        self.node_memory_size = 2 + random.choice([1, 2, 2, 3, 3, 4, 5])
        # 2 for sent_signal, sent_error
        return self.node_memory_size
    def sample_edge_memory_size(self):
        self.edge_memory_size = 4 + random.choice([1, 2, 2, 3, 3, 4, 5])
        # 4 for signal, has_signal, error, has_error
        return self.edge_memory_size

    def generate(self):
        self.sample_node_memory_size()
        self.sample_edge_memory_size()

        self.ruleset = self.ruleset_generator(node_memory_size=self.node_memory_size, edge_memory_size=self.edge_memory_size)
        self.ruleset.generate()

        self.model = Model()
        self.model.generate()

    def render(self):
        ruleset_code = self.ruleset.render()
        model_code = self.model.render()
        return ruleset_code + '\n\n\n' + model_code


class Model:
    # picks forward, backward, init, train, and async_train methods
    def __init__(self):
        pass

    def generate(self):
        # generating python code is a bit annoying because of indentation... don't assume indentation when generating each line; always add that later
        # of course this is really bad if we have any multiline strings...
        self.init_method = 'def __init__(self, input_size=2, output_size=1):pass'
        self.forward_method = 'def forward(self, *args, **kwargs):pass'
        self.backward_method = 'def backward(self, *args, **kwargs):pass'
        self.train_method = 'def train(self):pass'
        self.async_train_method = 'def async_train(self):pass'

        self.header = 'class Model:'

        self.methods = [self.init_method, self.forward_method, self.backward_method, self.train_method, self.async_train_method]


    def render(self):
        return self.header + '\n    ' + '\n    '.join(self.methods)


class Ruleset:
    def __init__(self, node_memory_size, edge_memory_size):
        self.node_memory_size = node_memory_size
        self.edge_memory_size = edge_memory_size
        self.number_filters = None

    # these methods are defined like this so they can be easily overridden by a subclass
    def sample_rules_per_slot(self, slot_type):
        assert slot_type in ['vector', 'float']
        if slot_type == 'vector':
            return random.choice([1, 1, 1, 1, 1, 2, 2, 3, 4, 5])
        if slot_type == 'float':
            return 1
    def sample_number_conditionals(self):
        self.number_conditionals = random.choice(list(range(4, 7)) + list(range(2, 9)))
        return self.number_conditionals
    def sample_conditional_complexities(self):
        if self.number_conditionals is None:
            raise Exception('must call sample_number_conditionals and assign self.number_conditionals before calling sample_conditional_complexities')
        self.conditional_complexities = [random.choice(list(range(1,5)) + list(range(1,11))) for i in range(self.number_conditionals)]
        return self.filter_complexities
    def sample_number_filters(self):
        self.number_filters = random.choice(list(range(4, 7)) + list(range(2, 9)))
        return self.number_filters
    def sample_filter_complexities(self):
        if self.number_filters is None:
            raise Exception('must call sample_number_filters and assign self.number_filters before calling sample_filter_complexities')
        self.filter_complexities = [random.choice(list(range(1,5)) + list(range(1,11))) for i in range(self.number_filters)]
        return self.filter_complexities
    def sample_step_expression_complexity(self, slot_type):
        assert slot_type in ['vector', 'float']
        if slot_type == 'vector':
            return random.choice(list(range(2,6)) + list(range(5,11))*3 + list(range(5,21))*2 + list(range(5,31)))
        if slot_type == 'float':
            return random.choice(list(range(2,6)) + list(range(5,11))*3)
    def sample_initialize_expression_complexity(self, slot_type):
        assert slot_type in ['vector', 'float']
        return random.choice([1, 2, 3])
    def sample_slot_filter_usage_frequency(self):
        self.slot_filter_usage_frequency = random.random()
        return self.slot_filter_usage_frequency
    def sample_slot_conditional_usage_frequency(self):
        self.slot_conditional_usage_frequency = random.random()
        return self.slot_conditional_usage_frequency
    def sample_base_expression_usage_frequency(self):
        self.base_expression_usage_frequency = random.random() * 0.25
        return self.base_expression_usage_frequency
    def sample_filter_break_frequency(self):
        self.filter_break_frequency = random.random() * 0.33
        return self.filter_break_frequency

    def generate(self):
        self.sample_number_filters()
        self.sample_filter_complexities()

        self.sample_number_conditionals()
        self.sample_conditional_complexities()

        self.sample_slot_filter_usage_frequency()
        self.sample_slot_conditional_usage_frequency()

        self.sample_base_expression_usage_frequency()
        self.sample_filter_break_frequency()

        self.filters = []

        # TODO: need default filters for signals - has_signal, has_error, not_has_signal, not_has_error
        for filter_complexity in self.filter_complexities:
            filter_expression = ExpressionTree(slot_type='vector', slot_filter=None, expression_complexity=filter_complexity, base_expression=None, expression_type='filter', ruleset=self, parent=None)
            filter_expression.generate()
            self.filters.append(filter_expression)

        self.conditionals = []
        # TODO: need default conditionals for signals - any edge has signal, any edge has error, no edge has signal, no edge has error; combined with sent_signal + sent_error
        for conditional_complexity in self.conditional_complexities:
            conditional_expression = ExpressionTree(slot_type='float', slot_filter=None, expression_complexity=conditional_complexity, base_expression=None, expression_type='conditional', ruleset=self, parent=None)
            conditional_expression.generate()
            self.conditionals.append(conditional_expression)

        slot_values = ['slot_node_{}'.format(i) for i in range(self.node_memory_size)] + ['slot_edge_{}'.format(i) for i in range(self.edge_memory_size)]
        slot_types = ['float'] * self.node_memory_size + ['vector'] * (self.edge_memory_size)
        slot_base_expressions = [leaves.context_renderer(leaves.node_memory, 'step', node_index=i) for i in range(self.node_memory_size)] + [leaves.context_renderer(leaves.edge_memory, 'step', edge_index=i) for i in range(self.edge_memory_size)]

        self.initialize_rules = []

        for slot_value, slot_type in zip(slot_values, slot_types):
            rule = Rule(ruleset=self, slot_type=slot_type, slot_value=slot_value, expression_complexity=self.sample_initialize_expression_complexity(slot_type), slot_filter_usage_frequency=0.0,  slot_conditional_usage_frequency=0.0, base_expression='leaves.zero()', expression_type='initialize')
            rule.generate()
            self.initialize_rules.append(rule)

        self.step_rules = []

        for slot_value, slot_type, slot_base_expression in zip(slot_values, slot_types, slot_base_expressions):
            self.step_rules.append([])
            for _ in range(self.sample_rules_per_slot(slot_type)):
                rule = Rule(ruleset=self, slot_type=slot_type, slot_value=slot_value, expression_complexity=self.sample_step_expression_complexity(slot_type), slot_filter_usage_frequency=self.slot_filter_usage_frequency, slot_conditional_usage_frequency=self.slot_conditional_usage_frequency, base_expression=slot_base_expression, expression_type='step')
                rule.generate()
                self.step_rules[-1].append(rule)

        return self

    def render(self):
        initialize_rules = self.initialize_rules
        step_rules = sum(self.step_rules, [])
        # turn these into code

        initialize_rule_lines = ['node_buffer = np.zeros(node_memory_size + edge_memory_size * edges)'] + [self.render_rule(initialize_rule_line) for initialize_rule_line in initialize_rules]
        initialize_rule = 'def initialize_rule(node_memory_size, edge_memory_size, edges):\n    {}'.format('\n    '.join(initialize_rule_lines))

        step_rule_lines = ['node_write_buffer[:] = node_read_buffer'] + [self.render_rule(step_rule_line) for step_rule_line in step_rules]
        step_rule = 'def step_rule(node_read_buffer, node_write_buffer, node_memory_size, edge_memory_size, edges):\n    {}'.format('\n    '.join(step_rule_lines))

        code = initialize_rule + '\n\n\n' + step_rule

        return code

    def render_rule(self, rule):
        assert rule.slot_type in ['vector', 'float']
        if rule.slot_type == 'vector' and rule.slot_filter is not None:
                return '(VECTOR) {slot_value}[filter={slot_filter}] = {rendered_expression}'.format(slot_value=rule.slot_value, slot_filter=self.render_expression_tree(rule.slot_filter), rendered_expression=self.render_expression_tree(rule.expression_tree))
        return '({slot_type}) {slot_value} = {rendered_expression}'.format(slot_type=rule.slot_type.upper(), slot_value=rule.slot_value, rendered_expression=self.render_expression_tree(rule.expression_tree))

    def render_expression_tree(self, expression_tree):
        # do it here because there's a lot of state which we really don't want the rules to also be keeping track of, which tells us, e.g, how many memory slots there are, etc
        if len(expression_tree.children) == 0:
            return expression_tree.operator
        else:
            return '{operator}({args})'.format(operator=operators.render(expression_tree.operator), args=', '.join(self.render_expression_tree(child) for child in expression_tree.children))


class Rule:
    def __init__(self, ruleset, slot_type, slot_value, expression_complexity, slot_filter_usage_frequency, slot_conditional_usage_frequency, base_expression, expression_type):
        self.ruleset = ruleset

        self.filters = ruleset.filters
        self.conditionals = ruleset.conditionals

        self.edge_memory_size = ruleset.edge_memory_size
        self.node_memory_size = ruleset.node_memory_size

        self.slot_type = slot_type
        self.slot_value = slot_value
        self.expression_complexity = expression_complexity
        self.slot_filter_usage_frequency = slot_filter_usage_frequency
        self.slot_conditional_usage_frequency = slot_conditional_usage_frequency
        self.base_expression = base_expression
        self.expression_type = expression_type

        self.slot_filter = None
        self.slot_conditional = None
        self.expression_tree = None

    def generate(self):
        slot_filter = None
        if self.filters and len(self.filters) and self.slot_type == 'vector' and random.random() < self.slot_filter_usage_frequency:
            # use a slot filter
            # (meaning this rule does not overwrite the whole slot; only some filtered version of the slot)
            slot_filter = random.choice(self.filters)
        self.slot_filter = slot_filter

        slot_conditional = None
        if self.conditionals and len(self.conditionals) and random.random() < self.slot_conditional_usage_frequency:
            # use a slot conditional
            # (meaning this rule is only run if this conditional is true)
            slot_conditional = random.choice(self.conditionals)
        self.slot_conditional = slot_conditional

        self.expression_tree = ExpressionTree(slot_type=self.slot_type, slot_filter=self.slot_filter, ruleset=self.ruleset, expression_complexity=self.expression_complexity, base_expression=self.base_expression, expression_type=self.expression_type, parent=None)
        self.expression_tree.generate()
        return self


class ExpressionTree:
    def __init__(self, slot_type, slot_filter, expression_complexity, base_expression, expression_type, ruleset, parent):
        self.slot_type = slot_type
        assert slot_type in ['vector', 'float']
        self.slot_filter = slot_filter
        # together the above two define the return type of this expression; a slot_filter does not make much sense on slot_type == float
        if self.slot_type != 'vector':
            assert self.slot_filter is None

        self.expression_complexity = expression_complexity
        # the target expression_complexity: complexity is for now defined as number of nodes in the ExpressionTree, but this may be changed to ignore certain kinds of nodes ()
        assert expression_complexity > 0

        self.base_expression = base_expression
        # the expression which this tree starts from; in case that needs to be overwritten (e.g, initialize rules default to a base_expression of 0, whereas step rules default to a base_expression of what is being assigned)

        self.expression_type = expression_type
        assert expression_type in ['initialize', 'step', 'filter', 'conditional']
        self.filters = ruleset.filters
        self.ruleset = ruleset
        self.edge_memory_size = ruleset.edge_memory_size
        self.node_memory_size = ruleset.node_memory_size

        if parent is None:
            self.parent = None
            self.root = self
            self.tree_depth = 1
        else:
            self.parent = parent
            self.root = parent.root
            self.tree_depth = 1
            traversed_node = self
            while traversed_node != self.root:
                traversed_node = traversed_node.parent
                self.tree_depth += 1

        self.children = []
        self.operator = None

    def generate(self):
        while self.get_complexity() < self.expression_complexity or self.get_complexity() > self.expression_complexity + 4:
            # add some buffer so we don't have to get the exact complexity amount
            # TODO: try changing that 4 to a 0 and see if it still runs in a reasonable amount of time or if there are pathological cases - WARNING - you need to implement decrease_complexity first
            if self.get_complexity() > self.expression_complexity:
                self.decrease_complexity()
            else:
                self.increase_complexity()
        return self

    def traverse_nodes(self):
        queue = [self]
        seen = set()
        while len(queue):
            node = queue.pop()
            seen.add(node)
            for child in node.children:
                if child not in seen:
                    queue.append(child)
                # else we may be dealing with either a cycle or child sharing (i.e, multiple parents)
                # cycles are bad but child sharing may be fine at some point in the future
            yield node

    def get_complexity(self):
        if self.operator is None:
            # uninitialized ExpressionTree has complexity=0
            return 0
        # below line includes self, so leaf complexity is 1, not 0
        return len(list(self.traverse_nodes()))

    def increase_complexity(self):
        # if we have no children and expression_complexity is 1-3, we know what kind of node we need to be
        # if expression_complexity is higher than 3, we need to select a random leaf node, delete it, and replace it with a node of complexity=2 or 3
        # along the way we also need to handle changing filters, picking types, constraining operators by type

        def adjust_node(number_children, node=self, child_complexity=1):
            # initializes this node and gives it this many children, each with complexity of 1 (i.e, a leaf)
            # 0 children = make it a leaf
            assert number_children in [0, 1, 2]

            assert node.operator is None or len(node.children) == 0
            # for now, let's not try adjusting nodes with children since keeping track of references is tricky
            # but it's ok to take a leaf node and give it children

            # assertions make sure this node hasn't been somehow already initialized
            if number_children == 0:
                leaf_type = node.slot_type
                if leaf_type is None:
                    leaf_type = random.choice(['vector', 'vector', 'float'])
                base_expression_type = node.root.slot_type
                if node.base_expression is not None and base_expression_type == leaf_type and random.random() < node.ruleset.base_expression_usage_frequency:
                    node.operator = node.base_expression
                else:
                    node.operator = leaves.sample_rendered_leaf(node.expression_type, leaf_type, self.node_memory_size, self.edge_memory_size)
            else:
                is_reducer = None
                child_slot_types = [random.choice(['vector', 'vector', 'float']) for _ in range(number_children)]
                node.operator = operators.sample_operator(is_reducer=is_reducer, number_children=number_children)
                if operators.get_is_reducer(node.operator):
                    child_slot_types = ['vector']

                child_slot_filters = []
                for child_slot_type in child_slot_types:
                    if child_slot_type == 'float':
                        child_slot_filters.append(None)
                    else:
                        if node.slot_type == 'vector':
                            child_slot_filter = node.slot_filter
                            if node.slot_filter is None:
                                # with some random chance give child a filter
                                # this means the child's return value will get "blown up" into the full vector length after being returned by the child
                                if random.random() < node.ruleset.filter_break_frequency:
                                    child_slot_filter = random.choice(node.filters)
                            else:
                                # with some random chance, remove the filter from the child: so this parent op when receiving the input from the child will actually select from a larger vector and drop most of the numbers
                                if random.random() < node.ruleset.filter_break_frequency:
                                    child_slot_filter = None
                            child_slot_filters.append(child_slot_filter)
                        else:
                            # we get to pick a slot filter!
                            slot_filter = None
                            if random.random() > 0.33:
                                if random.random() < 0.5:
                                    traversed_node = node
                                    while traversed_node.parent is not None:
                                        if traversed_node.slot_filter is not None:
                                            slot_filter = traversed_node.slot_filter
                                            break
                                        traversed_node = traversed_node.parent
                                if slot_filter is None and node.filters:
                                    slot_filter = random.choice(node.filters)
                            else:
                                # use no slot_filter
                                pass
                            child_slot_filters.append(slot_filter)
                for child_slot_type, child_slot_filter in zip(child_slot_types, child_slot_filters):
                    child_node = ExpressionTree(slot_type=child_slot_type, slot_filter=child_slot_filter, expression_complexity=child_complexity, base_expression=node.base_expression, expression_type=node.expression_type, ruleset=node.ruleset, parent=node)
                    child_node.generate()
                    node.children.append(child_node)


        if self.operator == None:
            # this is when we initialize an ExpressionTree
            # we don't want to create unnecessary complexity; and we want to be able to control from outside how deep it is when it gets initialized
            if self.expression_complexity in [1, 2]:
                # either this node needs to be a leaf, or a single child operator
                adjust_node(self.expression_complexity - 1)
            elif self.expression_complexity == 3:
                # we have to be either a single child operator with a child which is a single child operator which has a leaf node, or a two child operator with two leaf nodes
                number_children = random.choice([1, 2])
                if number_children == 1:
                    child_complexity = 2
                else:
                    child_complexity = 1
                adjust_node(number_children=number_children, child_complexity=child_complexity)
            else:
                # pick randomly out of leaf, single child operator, double child operator
                number_children = random.choice([0, 1, 2])
                adjust_node(number_children=number_children)
        else:
            # this node already was defined previously, and now we want to increase its complexity
            # if its a leaf node, turn it into an operator node - once this happens we should just ignore or adjust its expression_complexity
            # if its an operator node, call this method recursively on one of the leaf nodes
            is_leaf_node = len(self.children) == 0
            if is_leaf_node:
                number_children = random.choice([1, 2])
                child_complexity = random.choice([1, 2])
                adjust_node(number_children=number_children, child_complexity=child_complexity)
            else:
                descendants = list(self.traverse_nodes())
                leaf_nodes = [node for node in descendants if len(node.children) == 0]

                # we need to weight sampling by depth of the descendants, otherwise we have very high likelihood of unbalanced trees
                leaf_weights = [1.e9**(-leaf_node.tree_depth) for leaf_node in leaf_nodes]
                leaf_weights = np.array(leaf_weights) / sum(leaf_weights)

                node_to_adjust = np.random.choice(leaf_nodes, 1, p=leaf_weights)[0]

                node_to_adjust.increase_complexity()

    def decrease_complexity(self):
        # for now not necessary
        raise Exception(NotImplemented)
