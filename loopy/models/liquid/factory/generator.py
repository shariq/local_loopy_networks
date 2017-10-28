import random
from collections import defaultdict

# the way we make an expression of some complexity is to take a blank expression and randomly distort it, and keep doing that until we get our desired length
# increase_complexity should sample from the right distribution; so adding one node which may increase complexity by too much at once; but it's all approximate anyways so it's fine if it's a bit off

class Model:
    def __init__(self, ruleset_generator=None, harness_generator=None):
        if ruleset_generator is None:
            ruleset_generator = Ruleset
        self.ruleset_generator = ruleset_generator
        if harness_generator is None:
            harness_generator = Harness
        self.harness_generator = harness_generator

    def sample_node_memory_size(self):
        self.node_memory_size = random.choice([1, 2, 3, 4, 5])
        return self.node_memory_size
    def sample_edge_memory_size(self):
        self.edge_memory_size = random.choice([1, 2, 3, 4, 5])
        return self.edge_memory_size

    def generate(self):
        self.sample_node_memory_size()
        self.sample_edge_memory_size()

        self.ruleset = self.ruleset_generator(node_memory_size=self.node_memory_size, edge_memory_size=self.edge_memory_size)
        self.ruleset.generate()

        self.harness = self.harness_generator()
        self.harness.generate()

    def render(self):
        ruleset_code = self.ruleset.render()
        harness_code = self.harness.render()
        return ruleset_code + '\n\n\n' + harness_code


class Harness:
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

        self.header = 'class Harness:'

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

    def generate(self):
        self.sample_number_filters()
        self.sample_filter_complexities()
        self.sample_slot_filter_usage_frequency()
        filter_holder = FilterHolder(filter_complexities=self.filter_complexities)
        self.filters = filter_holder.generate_filters()

        slot_values = ['slot_node_{}'.format(i) for i in range(self.node_memory_size)] + ['slot_signal_0', 'slot_signal_1'] + ['slot_edge_{}'.format(i) for i in range(self.edge_memory_size)]
        slot_types = ['float'] * self.node_memory_size + ['vector'] * (2 + self.edge_memory_size)
        # 2 => signal + error vectors on each edge (signal_0 = signal; signal_1 = error)

        self.initialize_rules = []

        for slot_value, slot_type in zip(slot_values, slot_types):
            rule = Rule(filters=self.filters, slot_type=slot_type, slot_value=slot_value, expression_complexity=self.sample_initialize_expression_complexity(slot_type), slot_filter_usage_frequency=0.0, base_expression='float_0', is_initialize_rule=True)
            rule.generate()
            self.initialize_rules.append(rule)

        self.step_rules = []

        for slot_value, slot_type in zip(slot_values, slot_types):
            self.step_rules.append([])
            for _ in range(self.sample_rules_per_slot(slot_type)):
                rule = Rule(filters=self.filters, slot_type=slot_type, slot_value=slot_value, expression_complexity=self.sample_step_expression_complexity(slot_type), slot_filter_usage_frequency=self.slot_filter_usage_frequency, base_expression=slot_value, is_initialize_rule=False)
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
        if rule.slot_type == 'vector':
            return '(VECTOR) {slot_value}[filter={slot_filter}] = {rendered_expression}'.format(slot_value=rule.slot_value, slot_filter=rule.slot_filter, rendered_expression=self.render_expression_tree(rule.expression_tree))
        if rule.slot_type == 'float':
            return '(FLOAT) {slot_value} = {rendered_expression}'.format(slot_value=rule.slot_value, rendered_expression=self.render_expression_tree(rule.expression_tree))

    def render_expression_tree(self, expression_tree):
        # do it here because there's a lot of state which we really don't want the rules to also be keeping track of, which tells us, e.g, how many memory slots there are, etc
        if len(expression_tree.children) == 0:
            return expression_tree.operator
        else:
            return '{operator}({args})'.format(operator=expression_tree.operator, args=', '.join(self.render_expression_tree(child) for child in expression_tree.children))


class Rule:
    def __init__(self, filters, slot_type, slot_value, expression_complexity, slot_filter_usage_frequency, base_expression, is_initialize_rule):
        self.filters = filters
        self.slot_type = slot_type
        self.slot_value = slot_value
        self.expression_complexity = expression_complexity
        self.slot_filter_usage_frequency = slot_filter_usage_frequency
        self.base_expression = base_expression
        self.is_initialize_rule = is_initialize_rule
    def generate(self):
        slot_filter = None
        if self.slot_type == 'vector' and random.random() < self.slot_filter_usage_frequency:
            # use a slot filter
            # (meaning this rule does not overwrite the whole slot; only some filtered version of the slot)
            slot_filter = random.choice(self.filters)
        self.slot_filter = slot_filter
        self.expression_tree = ExpressionTree(slot_type=self.slot_type, slot_filter=self.slot_filter, filters=self.filters, expression_complexity=self.expression_complexity, base_expression=self.base_expression, is_initialize_rule=self.is_initialize_rule)
        self.expression_tree.generate()
        return self


class FilterHolder:
    def __init__(self, filter_complexities):
        self.filter_complexities = filter_complexities
    def generate_filters(self):
        filters = []
        for filter_complexity in self.filter_complexities:
            filters.append(self.generate_filter(filter_complexity=filter_complexity))
        self.filters = filters
        return filters
    def generate_filter(self, filter_complexity):
        filter_ = FilterExpressionTree(slot_type='vector', slot_filter=[], expression_complexity=filter_complexity, base_expression=None, is_initialize_rule=False, filters=[], parent=None)
        filter_.generate()
        return filter_


class ExpressionTree:
    def __init__(self, slot_type, slot_filter, expression_complexity, base_expression, is_initialize_rule, filters, parent=None):
        self.slot_type = slot_type
        self.slot_filter = slot_filter
        # together the above two define the return type of this expression; a slot_filter does not make much sense on slot_type == float
        if self.slot_type != 'vector':
            assert self.slot_filter is None

        self.expression_complexity = expression_complexity
        # the target expression_complexity: complexity is for now defined as number of nodes in the ExpressionTree, but this may be changed to ignore certain kinds of nodes ()

        self.base_expression = base_expression
        # the expression which this tree starts from; in case that needs to be overwritten (e.g, initialize rules default to a base_expression of 0, whereas update rules default to a base_expression of what is being assigned)

        self.is_initialize_rule = is_initialize_rule
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
                else:
                    raise Exception("this ExpressionTree was not a DAG; it had a cycle!")
            yield node

    def get_complexity(self):
        return len(list(self.traverse_nodes()))

    def increase_complexity(self):
        nodes = list(self.traverse_nodes())
        childless_nodes = [node for node in nodes if len(node.children) < 2]
        node_to_distort = random.choice(childless_nodes)
        node_to_distort.children.append(ExpressionTree(slot_type=self.slot_type, slot_filter=self.slot_filter, expression_complexity=1, base_expression=None, is_initialize_rule=self.is_initialize_rule, filters=self.filters, parent=self))

    def decrease_complexity(self):
        nodes = list(self.traverse_nodes())
        leaf_nodes = [node for node in nodes if len(node.children) == 0]
        node_to_delete = random.choice(leaf_nodes)
        node_to_delete.parent.children.remove(node_to_delete)


class FilterExpressionTree(ExpressionTree):
    pass
