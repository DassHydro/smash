import os, sys, re, pydoc
import sphinx
import inspect
import collections
import textwrap
import warnings
import smash

if sphinx.__version__ < '1.0.1':
    raise RuntimeError("Sphinx 1.0.1 or newer is required")

from numpydoc.numpydoc import mangle_docstrings
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList
from sphinx.domains.python import PythonDomain
from scipy._lib._util import getfullargspec_no_self


def setup(app):
    app.add_domain(SmashMinimize)
    return {'parallel_read_safe': True}


def _option_required_str(x):
    if not x:
        raise ValueError("value is required")
    return str(x)


def _import_object_interface(name):
    parts = name.split('.')
    module_name = "smash.core.model.Model"
    obj = getattr(eval(module_name), parts[-1])
    return obj

def _import_object_implementation(name):
    parts = name.split('.')
    module_name = ".".join(parts[:-1])
    obj = getattr(eval(module_name), parts[-1])
    return obj


class SmashMinimize(PythonDomain):
    name = 'smash-optimize'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.directives = dict(self.directives)
        self.directives['function'] = wrap_mangling_directive(self.directives['function'])


BLURB = """
.. seealso:: For documentation for the rest of the parameters, see `%s`
"""

BLURB_NELDER_MEAD = """
.. seealso:: For documentation for the rest of the parameters, see `%s` 
\n For documentation for the rest of the options, see SciPy minimize documentation `here <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html#optimize-minimize-neldermead>`__
"""

BLURB_L_BFGS_B = """
.. seealso:: For documentation for the rest of the parameters, see `%s` 
\n For documentation for the rest of the options, see SciPy minimize documentation `here <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb>`__
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
            impl_name = self.options['impl']
            impl_obj = _import_object_implementation(impl_name)
            impl_args, impl_varargs, impl_keywords, impl_defaults = getfullargspec_no_self(impl_obj)[:4]

            # Format signature taking implementation into account
            args = list(args)
            defaults = [None] + list(defaults)

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
            rules_ind = args.index("rules")
            impl_flag = 0
            for j, opt_name in enumerate(impl_args):
                if opt_name in args:
                    continue
                    
                if j >= rules_ind:

                    if impl_flag <= len(impl_defaults):
                        options.append((opt_name, impl_defaults[impl_flag]))
                        impl_flag += 1

                    else:
                        options.append((opt_name, None))

            set_default('options', dict(options))
            
            if 'method' in self.options and 'method' in args:
                set_default('method', self.options['method'].strip())
            elif 'solver' in self.options and 'solver' in args:
                set_default('solver', self.options['solver'].strip())

            special_args = {'solver', 'obj_fun', 'method', 'rules', 'inplace', 'options'}
            for arg in list(args):
                if arg not in impl_args and arg not in special_args:
                    remove_arg(arg)

            # XXX deprecation that we should fix someday using Signature (?)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('ignore')
                signature = inspect.formatargspec(
                    args, varargs, keywords, defaults)

            # Produce output
            self.options['noindex'] = True
            self.arguments[0] = name + signature
            lines = textwrap.dedent(pydoc.getdoc(impl_obj)).splitlines()
            # Change "Options" to "Other Parameters", run numpydoc, reset
            new_lines = []
            for line in lines:
                # Remap Options to the "Other Parameters" numpydoc section
                # along with correct heading length
                if line.strip() == 'Options':
                    line = "Other Parameters"
                    new_lines.extend([line, "-"*len(line)])
                    continue
                new_lines.append(line)
            # use impl_name instead of name here to avoid duplicate refs
            mangle_docstrings(env.app, 'function', impl_name,
                              None, None, new_lines) 
            lines = new_lines
            new_lines = []
            for line in lines:
                if line.strip() == ':Other Parameters:':
                    
                    if self.options['solver'] in 'nelder-mead':
                        new_lines.extend((BLURB_NELDER_MEAD % (name,)).splitlines())
                        
                    else:
                        new_lines.extend((BLURB % (name,)).splitlines())
                    
                    new_lines.append('\n')
                    new_lines.append(':Options:')
                    
                elif line.strip() == '**-------**':
                    pass
                    
                else:
                    new_lines.append(line)
            self.content = ViewList(new_lines, self.content.parent)
            return base_directive.run(self)

        option_spec = dict(base_directive.option_spec)
        option_spec['impl'] = _option_required_str
        option_spec['solver'] = _option_required_str

    return directive
