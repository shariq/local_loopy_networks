import argparse
import local_learning
import logging
from types import ModuleType

logger = logging.getLogger()
debug_exceptions = local_learning.debug_exceptions

from local_learning.models.loopy.checks import all_checks, all_checks_accuracy_requirements
import local_learning.models.loopy.factory.generator as generator
from local_learning.models.backprop import BackpropModel


def debug_log_code(code):
    lines = code.splitlines()
    fill_size = 1 + len(str(len(lines)))
    new_lines = ['{l}|{c}'.format(l=str(i+1).ljust(fill_size), c=line) for i, line in enumerate(lines)]
    return '\n'.join(new_lines)


def compile_model(harness_code, class_name='Model', module_name='harness_module'):
    if logger.level >= logging.DEBUG:
        # prevent annoying overhead
        logger.debug(debug_log_code(harness_code))
    compiled = compile(harness_code, '<string>', 'exec')
    module = ModuleType(module_name)
    exec(compiled, module.__dict__)
    return module.__dict__[class_name]


def sample_backprop_model_class():
    generator_harness = None
    harness_code = 'backprop_code'
    model_class = BackpropModel
    return generator_harness, harness_code, model_class


def sample_factory_model_class():
    generator_harness = generator.Harness()
    generator_harness.generate()
    harness_code = generator_harness.render()
    model_class = compile_model(harness_code)
    return generator_harness, harness_code, model_class


def search_harness(model_class_sampler=sample_factory_model_class, limit=None, hardcoded_results=False):
    iterations = 0
    while True:
        if limit is not None and iterations >= limit:
            break
        try:
            generator_harness, harness_code, model_class = model_class_sampler()
            results = [0.0] * len(all_checks)
            if hardcoded_results:
                yield generator_harness, harness_code, results
            else:
                for check_index, check, accuracy_requirement in zip(range(len(all_checks)), all_checks, all_checks_accuracy_requirements):
                    try:
                        check_accuracy = check(model_class)
                        results[check_index] = check_accuracy
                        if check_accuracy < accuracy_requirement:
                            break
                    except Exception as e:
                        logger.error(e, exc_info=True)
                        if debug_exceptions:
                            import pdb ; pdb.set_trace()
                            raise
                        else:
                            break
                iterations += 1
                yield generator_harness, harness_code, results
        except Exception as e:
            iterations += 1
            logger.error(e, exc_info=True)
            if debug_exceptions:
                import pdb ; pdb.set_trace()
                raise
