# takes in a tree describing a spec of an initialize and step rule
# turns it into an initialize and step function

from types import ModuleType
import loopy.models.liquid.factory.generator as generator

def interpret(code):
    compiled = compile(code, '', 'exec')
    module = ModuleType('liquidmodule')
    exec(compiled, module.__dict__)
    return module.__dict__['Harness']()

def sample_harness():
    return interpret(generator.sample_model_code())
