import local_learning.models.loopy.factory.leaves as leaves
import local_learning.models.loopy.factory.operators as operators



def indent_code_block(code_block):
    return '    \n'.join(code_block.splitlines())


class Renderer:
    def __init__(self, harness):
        self.harness = harness
        self.ruleset = harness.ruleset
        self.model = harness.model
        self.node_memory_size = harness.node_memory_size
        self.edge_memory_size = harness.edge_memory_size


    def render(self):
        ruleset_code = self.render_ruleset()
        model_code = self.render_model()
        return ruleset_code + '\n\n\n' + model_code


    def render_ruleset(self):
        ruleset = self.ruleset

        initialize_rules = ruleset.initialize_rules
        step_rules = sum(ruleset.step_rules, [])
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


    def render_model(self):
        model = self.model
        header = model.header
        methods = model.methods

        format_arguments = {
            'edge_memory_size': self.edge_memory_size,
            'node_memory_size': self.node_memory_size,
            'adjacency_dict': model.adjacency_dict,
            'NODE_SIGNAL_MEMORY_INDEX': 0,
            'NODE_SIGNAL_JUST_SENT_INDEX': 1,
            'NODE_ERROR_JUST_SENT_INDEX': 2,
            'EDGE_SIGNAL_INDEX': 0,
            'EDGE_HAS_SIGNAL_INDEX': 1,
            'EDGE_ERROR_INDEX': 2,
            'EDGE_HAS_ERROR_INDEX': 3,
            'EDGE_WEIGHT_INDEX': 4
        }

        rendered_methods = '\n\n'.join([indent_code_block(method.format(**format_arguments)).strip() for method in methods])

        code = '{header}\n{rendered_methods}'.format(header=header, rendered_methods=rendered_methods)

        return code
