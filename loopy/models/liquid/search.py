import loopy
import logging
logger = logging.getLogger()

from loopy.models.liquid.checks import all_checks, all_checks_accuracy_requirements

from types import ModuleType
import loopy.models.liquid.factory.generator as generator


def compile_harness(harness_code, class_name='Harness', module_name='harness_module'):
    print(harness_code)
    compiled = compile(harness_code, '', 'exec')
    module = ModuleType(module_name)
    exec(compiled, module.__dict__)
    return module.__dict__[class_name]


def search_harness():
    while True:
        try:
            generator_model = generator.Model()
            generator_model.generate()
            harness_code = generator_model.render()
            harness_class = compile_harness(harness_code)

            #generator_model = None
            #harness_code = 'backprop'
            #from loopy.models.backprop import BackpropModel as harness_class
            ## above tests that backprop as a harness does in fact work with these checks
        except Exception as e:
            logger.error(e, exc_info=True)
            continue
        results = [0.0] * len(all_checks)
        for check_index, check, accuracy_requirement in zip(range(len(all_checks)), all_checks, all_checks_accuracy_requirements):
            try:
                check_accuracy = check(harness_class)
                if check_accuracy >= accuracy_requirement:
                    results[check_index] = check_accuracy
                else:
                    break
            except Exception as e:
                logger.error(e, exc_info=True)
                break
        yield generator_model, harness_code, results


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    for _, code, results in search_harness():
        total_score = sum(results) / len(results)
        logger.info('{} got score {} with results {}'.format(code[:200].replace('\n', '\\n'), total_score, results))
