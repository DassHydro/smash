# ruff: noqa

import pydoc
import sphinx
import inspect
import textwrap
import warnings
import smash

if sphinx.__version__ < "1.0.1":
    raise RuntimeError("Sphinx 1.0.1 or newer is required")

from numpydoc.numpydoc import mangle_docstrings
from docutils.statemachine import ViewList
from sphinx.domains.python import PythonDomain
from scipy._lib._util import getfullargspec_no_self


def setup(app):
    app.add_domain(SmashModelOptimize)
    app.add_domain(SmashNetAdd)
    app.add_domain(SmashNetCompile)
    return {"parallel_read_safe": True}


def _option_required_str(x):
    if not x:
        raise ValueError("value is required")
    return str(x)


def _import_object_interface(name):
    parts = name.split(".")
    if parts[0] == "Model":
        module_name = "smash.core.model.model.Model"
    elif parts[0] == "Net":
        module_name = "smash.factory.net.net.Net"
    else:
        raise ValueError(f"Unknown object interface {name}")
    obj = getattr(eval(module_name), parts[-1])
    return obj


def _import_object_implementation(name):
    parts = name.split(".")
    module_name = ".".join(parts[:-1])
    obj = getattr(eval(module_name), parts[-1])
    return obj


class SmashModelOptimize(PythonDomain):
    name = "smash-model-optimize"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.directives = dict(self.directives)
        self.directives["function"] = wrap_mangling_directive(self.directives["function"])


class SmashNetAdd(PythonDomain):
    name = "smash-net-add"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.directives = dict(self.directives)
        self.directives["function"] = wrap_mangling_directive(self.directives["function"])


class SmashNetCompile(PythonDomain):
    name = "smash-net-compile"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.directives = dict(self.directives)
        self.directives["function"] = wrap_mangling_directive(self.directives["function"])


BLURB = """
.. seealso:: For documentation for the rest of the parameters, see `%s`
"""


def wrap_mangling_directive(base_directive):
    class directive(base_directive):
        def run(self):
            env = self.state.document.settings.env

            # Interface function
            name = self.arguments[0].strip()
            obj = _import_object_interface(name)
            args, varargs, keywords, defaults = getfullargspec_no_self(obj)[:4]

            # Implementation function
            impl_name = self.options["impl"]
            impl_obj = _import_object_implementation(impl_name)
            (
                impl_args,
                impl_varargs,
                impl_keywords,
                impl_defaults,
            ) = getfullargspec_no_self(impl_obj)[:4]

            # Format signature taking implementation into account
            args = list(args)

            if defaults is not None:  # only remove args and set default if defaults is not None
                defaults = list(defaults)

                def set_default(arg, value):
                    j = args.index(arg)
                    defaults[len(defaults) - (len(args) - j)] = value

                def remove_arg(arg):
                    if arg not in args:
                        return
                    j = args.index(arg)
                    if j < len(args) - len(defaults):
                        del args[j]
                    else:
                        del defaults[len(defaults) - (len(args) - j)]
                        del args[j]

                options = []
                for j, opt_name in enumerate(impl_args):
                    if opt_name in args:
                        continue
                    if j >= len(impl_args) - len(impl_defaults):
                        options.append(
                            (
                                opt_name,
                                impl_defaults[len(impl_defaults) - (len(impl_args) - j)],
                            )
                        )
                    else:
                        options.append((opt_name, None))

                set_default("options", dict(options))

                if "algorithm" in self.options and "algorithm" in args:
                    set_default("algorithm", self.options["algorithm"].strip())
                elif "alg" in self.options and "alg" in args:
                    set_default("alg", self.options["alg"].strip())

                if "optimizer" in self.options and "optimizer" in args:
                    set_default("optimizer", self.options["optimizer"].strip())
                elif "opt" in self.options and "opt" in args:
                    set_default("opt", self.options["opt"].strip())

                special_args = set(inspect.getfullargspec(smash.Model.optimize).args)

                for arg in list(args):
                    if arg not in impl_args and arg not in special_args:
                        remove_arg(arg)

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")
                signature = inspect.signature(obj)
                mangled_signature = []
                for parameters in signature.parameters.values():
                    if parameters.name == "self":
                        continue
                    elif parameters.name == "mapping" and self.options["alg"] == "l-bfgs-b":
                        default = "'distributed'"
                    elif parameters.name == "optimizer":
                        default = f"'{self.options['opt']}'"
                    else:
                        if isinstance(parameters.default, str):
                            default = f"'{parameters.default}'"
                        else:
                            default = parameters.default

                    if defaults is None:
                        mangled_signature.append(f"{parameters.name}")
                    else:
                        mangled_signature.append(f"{parameters.name}={default}")
                mangled_signature = f"({', '.join(mangled_signature)})"

            # Produce output
            self.options["noindex"] = True
            self.arguments[0] = name + mangled_signature
            lines = textwrap.dedent(pydoc.getdoc(impl_obj)).splitlines()

            # Change "Options" to "Other Parameters", run numpydoc, reset
            new_lines = []
            for line in lines:
                # Remap Options to the "Other Parameters" numpydoc section
                # along with correct heading length
                if line.strip() == "Options":
                    line = "Other Parameters"
                    new_lines.extend([line, "-" * len(line)])
                    continue
                new_lines.append(line)

            # use impl_name instead of name here to avoid duplicate refs
            mangle_docstrings(env.app, "function", impl_name, None, None, new_lines)
            lines = new_lines
            new_lines = []
            for line in lines:
                if line.strip() == ":Other Parameters:":
                    new_lines.extend((BLURB % (name,)).splitlines())
                    new_lines.append("\n")
                    new_lines.append(":Options:")

                elif line.strip() in [
                    "**-------**",
                    "..",
                    "!! processed by numpydoc !!",
                ]:
                    pass

                else:
                    new_lines.append(line)
            if sphinx.__version__ >= "7.4":
                self.content = "\n".join(new_lines)
            else:
                self.content = ViewList(new_lines, self.content.parent)
            return base_directive.run(self)

        option_spec = dict(base_directive.option_spec)
        option_spec["impl"] = _option_required_str
        option_spec["alg"] = _option_required_str
        option_spec["opt"] = _option_required_str

    return directive
