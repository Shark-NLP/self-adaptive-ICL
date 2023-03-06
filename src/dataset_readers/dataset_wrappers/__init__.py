import glob
import pathlib
import types
from os.path import dirname, isfile, join

modules = {}
modules_list = glob.glob(join(dirname(__file__), "*.py"))
for path in modules_list:
    if isfile(path) and not path.endswith('__init__.py') and not path.endswith('task_.py'):
        mod_name = pathlib.Path(path).name[:-3]
        module = types.ModuleType(mod_name)
        with open(path, encoding='utf-8') as f:
            module_str = f.read()
        exec(module_str, module.__dict__)
        modules[mod_name] = module

task_list = {}
for module_name, module in modules.items():
    for el in dir(module):
        if el.endswith("DatasetWrapper"):
            obj = module.__dict__[el]
            task_list[obj.name] = obj


def get_dataset_wrapper(name):
    return task_list[name]
