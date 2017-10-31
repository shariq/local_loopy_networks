import random
from collections import defaultdict

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
        self.node_memory_size = 2 + random.choice([1, 2, 3, 4, 5])
        # 2 for sent_signal, sent_error
        return self.node_memory_size
    def sample_edge_memory_size(self):
        self.edge_memory_size = 4 + random.choice([1, 2, 3, 4, 5])
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
            return random.choice(list(range(1,6)) + list(range(1,11))*3 + list(range(1,21))*2 + list(range(1,31)))
        if slot_type == 'float':
            return random.choice(list(range(1,3)) + list(range(1,6)) + list(range(5,11))*3)
    def sample_initialize_expression_complexity(self, slot_type):
        assert slot_type in ['vector', 'float']
        return random.choice([1, 2, 3])
    def sample_slot_filter_usage_frequency(self):
        self.slot_filter_usage_frequency = random.random()
        return self.slot_filter_usage_frequency
    def sample_slot_conditional_usage_frequency(self):
        self.slot_conditional_usage_frequency = random.random()
        return self.slot_conditional_usage_frequency

    def generate(self):
        self.sample_number_filters()
        self.sample_filter_complexities()

        self.sample_number_conditionals()
        self.sample_conditional_complexities()

        self.sample_slot_filter_usage_frequency()
        self.sample_slot_conditional_usage_frequency()

        self.filters = []

        # TODO: need default filters for signals - has_signal, has_error, not_has_signal, not_has_error
        for filter_complexity in self.filter_complexities:
            filter_expression = ExpressionTree(slot_type='vector', slot_filter=None, expression_complexity=filter_complexity, base_expression=None, expression_type='filter', filters=[], parent=None)
            filter_expression.generate()
            self.filters.append(filter_expression)

        self.conditionals = []
        # TODO: need default conditionals for signals - any edge has signal, any edge has error, no edge has signal, no edge has error; combined with sent_signal + sent_error
        for conditional_complexity in self.conditional_complexities:
            conditional_expression = ExpressionTree(slot_type='float', slot_filter=None, expression_complexity=conditional_complexity, base_expression=None, expression_type='conditional', filters=self.filters, parent=None)
            conditional_expression.generate()
            self.conditionals.append(conditional_expression)

        slot_values = ['slot_node_{}'.format(i) for i in range(self.node_memory_size)] + ['slot_signal_0', 'slot_signal_1'] + ['slot_edge_{}'.format(i) for i in range(self.edge_memory_size)]
        slot_types = ['float'] * self.node_memory_size + ['vector'] * (2 + self.edge_memory_size)
        # 2 => signal + error vectors on each edge (signal_0 = signal; signal_1 = error)

        self.initialize_rules = []

        for slot_value, slot_type in zip(slot_values, slot_types):
            rule = Rule(filters=self.filters, conditionals=self.conditionals, slot_type=slot_type, slot_value=slot_value, expression_complexity=self.sample_initialize_expression_complexity(slot_type), slot_filter_usage_frequency=0.0,  slot_conditional_usage_frequency=0.0, base_expression='float_0', expression_type='initialize')
            rule.generate()
            self.initialize_rules.append(rule)

        self.step_rules = []

        for slot_value, slot_type in zip(slot_values, slot_types):
            self.step_rules.append([])
            for _ in range(self.sample_rules_per_slot(slot_type)):
                rule = Rule(filters=self.filters, conditionals=self.conditionals, slot_type=slot_type, slot_value=slot_value, expression_complexity=self.sample_step_expression_complexity(slot_type), slot_filter_usage_frequency=self.slot_filter_usage_frequency, slot_conditional_usage_frequency=self.slot_conditional_usage_frequency, base_expression=slot_value, expression_type='step')
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
            return '{operator}({args})'.format(operator=expression_tree.operator, args=', '.join(self.render_expression_tree(child) for child in expression_tree.children))


class Rule:
    def __init__(self, filters, conditionals, slot_type, slot_value, expression_complexity, slot_filter_usage_frequency, slot_conditional_usage_frequency, base_expression, expression_type):
        self.filters = filters
        self.conditionals = conditionals
        self.slot_type = slot_type
        self.slot_value = slot_value
        self.expression_complexity = expression_complexity
        self.slot_filter_usage_frequency = slot_filter_usage_frequency
        self.slot_conditional_usage_frequency = slot_conditional_usage_frequency
        self.base_expression = base_expression
        self.expression_type = expression_type

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

        self.expression_tree = ExpressionTree(slot_type=self.slot_type, slot_filter=self.slot_filter, filters=self.filters, expression_complexity=self.expression_complexity, base_expression=self.base_expression, expression_type=self.expression_type, parent=None)
        self.expression_tree.generate()
        return self


class ExpressionTree:
    def __init__(self, slot_type, slot_filter, expression_complexity, base_expression, expression_type, filters, parent):
        self.slot_type = slot_type
        assert slot_type in ['vector', 'float']
        self.slot_filter = slot_filter
        # together the above two define the return type of this expression; a slot_filter does not make much sense on slot_type == float
        if self.slot_type != 'vector':
            assert self.slot_filter is None

        self.expression_complexity = expression_complexity
        # the target expression_complexity: complexity is for now defined as number of nodes in the ExpressionTree, but this may be changed to ignore certain kinds of nodes ()

        self.base_expression = base_expression
        # the expression which this tree starts from; in case that needs to be overwritten (e.g, initialize rules default to a base_expression of 0, whereas step rules default to a base_expression of what is being assigned)

        self.expression_type = expression_type
        assert expression_type in ['initialize', 'step', 'filter', 'conditional']
        self.filters= filters

        if parent is None:
            self.parent = None
            self.root = self
        else:
            self.parent = parent
            self.root = parent.root

        self.children = []
        self.operator = 'None'

    def generate(self):
        if self.base_expression is not None:
            self.operator = self.base_expression
            # initialize ExpressionTree with this operator
        else:
            self.operator = random.choice(['add', 'multiply'])

        # do some stuff to ensure base_expression is valid: if it needs children, give it children

        while abs(self.get_complexity() - self.expression_complexity) >= 3:
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
        return len(list(self.traverse_nodes()))

    def increase_complexity(self):
        nodes = list(self.traverse_nodes())
        childless_nodes = [node for node in nodes if len(node.children) < 2]
        node_to_distort = random.choice(childless_nodes)
        new_node = ExpressionTree(slot_type=self.slot_type, slot_filter=self.slot_filter, expression_complexity=1, base_expression=None, expression_type=self.expression_type, filters=self.filters, parent=self)
        new_node.generate()
        node_to_distort.children.append(new_node)

    def decrease_complexity(self):
        nodes = list(self.traverse_nodes())
        leaf_nodes = [node for node in nodes if len(node.children) == 0]
        node_to_delete = random.choice(leaf_nodes)
        node_to_delete.parent.children.remove(node_to_delete)
