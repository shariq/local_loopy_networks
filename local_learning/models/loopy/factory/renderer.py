import local_learning.models.loopy.factory.leaves as leaves
import local_learning.models.loopy.factory.operators as operators

import local_learning
debug_exceptions = local_learning.debug_exceptions


def indent_code_block(code_block):
    return '\n'.join('    ' + line for line in code_block.splitlines())


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
        initialize_rule_code = self.render_initialize_rule_code()
        step_rule_code = self.render_step_rule_code()

        code = initialize_rule_code + '\n\n\n' + step_rule_code

        return code


    def render_initialize_rule_code(self):
        method_header = 'def initialize_rule(node_memory_size, edge_memory_size, edges):'

        edge_slot_assignments = '\n'.join([
            'slot_edge_{i}=node_write_buffer[{node_memory_size}+{i}::{edge_memory_size}]'.format(
                i=i,
                edge_memory_size=self.edge_memory_size,
                node_memory_size=self.node_memory_size
            ) for i in range(self.edge_memory_size)])

        ruleset_generated_code = '\n'.join(self.render_rule(initialize_rule) for initialize_rule in self.ruleset.initialize_rules)

        method_body = edge_slot_assignments.strip() + '\n' + ruleset_generated_code.strip()

        method_contents = '''node_write_buffer = np.zeros(node_memory_size + edge_memory_size * edges)
try:
    {method_body}
except Exception as e:
    {debug_exceptions_pdb}
    raise
return node_write_buffer'''.format(method_body=indent_code_block(method_body).strip(), debug_exceptions_pdb=('import pdb ; pdb.set_trace()' if debug_exceptions else ''))

        return '{method_header}\n{method_contents}'.format(method_header=method_header, method_contents=indent_code_block(method_contents))


    def render_step_rule_code(self):
        method_header = 'def step_rule(node_read_buffer, node_write_buffer, node_memory_size, edge_memory_size, edges):'

        edge_slot_assignments = '\n'.join([
            'slot_edge_{i}=node_write_buffer[{node_memory_size}+{i}::{edge_memory_size}]'.format(
                i=i,
                edge_memory_size=self.edge_memory_size,
                node_memory_size=self.node_memory_size
            ) for i in range(self.edge_memory_size)])

        filter_initialization = '\n'.join([
            'slot_filter_{i}={expression}>0.0'.format(
                i=i, expression=self.render_expression_tree(filter_i)
            ) for i, filter_i in enumerate(self.ruleset.filters)
        ])

        conditional_initialization = '\n'.join([
            'slot_conditional_{i}=bool({expression}>0.0)'.format(
                i=i, expression=self.render_expression_tree(conditional_i)
            ) for i, conditional_i in enumerate(self.ruleset.conditionals)
        ])

        ruleset_generated_code = '\n'.join(self.render_rule(step_rule_line) for step_rule_line in sum(self.ruleset.step_rules, []))
        # stripped so we can indent it nicely below

        method_body = edge_slot_assignments.strip() + '\n' + filter_initialization.strip() + '\n' + conditional_initialization.strip() + '\n' + ruleset_generated_code.strip()

        method_contents = '''node_write_buffer[:] = node_read_buffer
try:
    {method_body}
except Exception as e:
    logger.error(e, exc_info=True)
    {debug_exceptions_pdb}
    raise'''.format(method_body=indent_code_block(method_body).strip(), debug_exceptions_pdb=('import pdb ; pdb.set_trace()' if debug_exceptions else ''))

        return '{method_header}\n{method_contents}'.format(method_header=method_header, method_contents=indent_code_block(method_contents))


    def render_rule(self, rule):
        rendered_code = ''
        # all of this should fit on a single line; to keep life simple

        if rule.slot_conditional:
            conditional_index = self.ruleset.conditionals.index(rule.slot_conditional)
            rendered_code += 'if slot_conditional_{}:'.format(conditional_index)

        if rule.slot_type == 'vector':
            if rule.slot_filter is not None:
                rendered_code += '{slot_value}[slot_filter_{slot_filter_i}]='.format(slot_value=rule.slot_value, slot_filter_i=self.ruleset.filters.index(rule.slot_filter))
            else:
                rendered_code += '{slot_value}[:]='.format(slot_value=rule.slot_value)
        else:
            rendered_code += '{slot_value}='.format(slot_value=rule.slot_value)

        rendered_code += self.render_expression_tree(rule.expression_tree)

        return rendered_code


    def render_expression_tree(self, expression_tree):
        if len(expression_tree.children) == 0:
            rendered_expression = operators.render(expression_tree.operator)
        else:
            rendered_expression = '{operator}({args})'.format(operator=operators.render(expression_tree.operator), args=', '.join(self.render_expression_tree(child) for child in expression_tree.children))
        # need to do weird shit with filters...

        if expression_tree.slot_type == 'float':
            return rendered_expression

        node_slot_filter = expression_tree.slot_filter

        if expression_tree.slot_type == 'vector' and expression_tree.parent == None:
            if node_slot_filter is not None:
                node_slot_filter_i = self.ruleset.filters.index(node_slot_filter)
                return 'operators.ensure_vector({}, edges, slot_filter_{})'.format(rendered_expression, node_slot_filter_i)
            else:
                return 'operators.ensure_vector({}, edges, None)'.format(rendered_expression)

        parent_slot_filter = expression_tree.parent.slot_filter

        if parent_slot_filter == node_slot_filter:
            return rendered_expression

        if parent_slot_filter is None and node_slot_filter is not None:
            node_slot_filter_i = self.ruleset.filters.index(node_slot_filter)
            return 'operators.undo_filter({expression}, slot_filter_{i})'.format(expression=rendered_expression, i=node_slot_filter_i)

        if parent_slot_filter is not None and node_slot_filter is None:
            parent_slot_filter_i = self.ruleset.filters.index(parent_slot_filter)
            return 'operators.apply_filter({expression}, slot_filter_{i})'.format(expression=rendered_expression, i=parent_slot_filter_i)

        if parent_slot_filter is not None and node_slot_filter is not None:
            # ouch ; they're both different...
            parent_slot_filter_i = self.ruleset.filters.index(parent_slot_filter)
            node_slot_filter_i = self.ruleset.filters.index(node_slot_filter)
            return 'operators.apply_filter(operators.undo_filter({expression}, slot_filter_{node_i}), slot_filter_{parent_i})'.format(expression=rendered_expression, node_i=node_slot_filter_i, parent_i=parent_slot_filter_i)


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

        rendered_methods = '\n\n'.join([indent_code_block(method.format(**format_arguments).strip()) for method in methods])

        code = '{header}\n{rendered_methods}'.format(header=header, rendered_methods=rendered_methods)

        return code
