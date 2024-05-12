project('hydro-smash',
    'c',
    version: run_command(['generate_version.py', '--print'], check: true).stdout().strip(),
    license: 'GPL-3.0',
    meson_version: '>=1.2.1',
    default_options: [
        'warning_level=2',
        'buildtype=debugoptimized',
    ],
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

fs = import('fs')
versioneer = files('generate_version.py')

if fs.exists('_version_meson.py')
    py.install_sources('_version_meson.py', subdir: 'smash')
else
    custom_target('write_version_file',
        output: '_version_meson.py',
        command: [
            py, versioneer, '-o', '@OUTPUT@'
        ],
        build_by_default: true,
        build_always_stale: true,
        install: true,
        install_dir: py.get_install_dir() / 'smash'
    )
    meson.add_dist_script(py, versioneer, '-o', '_version_meson.py')
endif

cc = meson.get_compiler('c')

# Adding at project level causes many spurious -lgfortran flags.
add_languages('fortran', native: false)
ff = meson.get_compiler('fortran')

# C warning flags
wno_maybe_uninitialized = cc.get_supported_arguments('-Wno-maybe-uninitialized')
wno_discarded_qualifiers = cc.get_supported_arguments('-Wno-discarded-qualifiers')
wno_empty_body = cc.get_supported_arguments('-Wno-empty-body')
wno_implicit_function_declaration = cc.get_supported_arguments('-Wno-implicit-function-declaration')
wno_parentheses = cc.get_supported_arguments('-Wno-parentheses')
wno_switch = cc.get_supported_arguments('-Wno-switch')
wno_unused_label = cc.get_supported_arguments('-Wno-unused-label')
wno_unused_result = cc.get_supported_arguments('-Wno-unused-result')
wno_unused_variable = cc.get_supported_arguments('-Wno-unused-variable')
wno_unused_parameter = cc.get_supported_arguments('-Wno-unused-parameter')
wno_cast_function_type = cc.get_supported_arguments('-Wno-cast-function-type')
wno_missing_field_initializers = cc.get_supported_arguments('-Wno-missing-field-initializers')

c_ignore_warnings = ff.get_supported_arguments(
    wno_maybe_uninitialized,
    wno_unused_label,
    wno_unused_variable,
    wno_unused_parameter,
    wno_cast_function_type,
    wno_missing_field_initializers,
)

# Fortran warning flags
_fflag_wno_argument_mismatch = ff.get_supported_arguments('-Wno-argument-mismatch')
_fflag_wno_conversion = ff.get_supported_arguments('-Wno-conversion')
_fflag_wno_intrinsic_shadow = ff.get_supported_arguments('-Wno-intrinsic-shadow')
_fflag_wno_maybe_uninitialized = ff.get_supported_arguments('-Wno-maybe-uninitialized')
_fflag_wno_surprising = ff.get_supported_arguments('-Wno-surprising')
_fflag_wno_uninitialized = ff.get_supported_arguments('-Wno-uninitialized')
_fflag_wno_unused_dummy_argument = ff.get_supported_arguments('-Wno-unused-dummy-argument')
_fflag_wno_unused_label = ff.get_supported_arguments('-Wno-unused-label')
_fflag_wno_unused_variable = ff.get_supported_arguments('-Wno-unused-variable')
_flags_wno_unused_parameter = ff.get_supported_arguments('-Wno-unused-parameter')
_fflag_wno_tabs = ff.get_supported_arguments('-Wno-tabs')
_fflag_wno_compare_reals = ff.get_supported_arguments('-Wno-compare-reals')
_fflag_wno_character_truncation = ff.get_supported_arguments('-Wno-character-truncation')
_fflag_wno_surprising = ff.get_supported_arguments('-Wno-surprising')
# The default list of warnings to ignore from Fortran code. There is a lot of
# old, vendored code that is very bad and we want to compile it silently (at
# least with GCC and Clang)
fortran_ignore_warnings = ff.get_supported_arguments(
    _fflag_wno_argument_mismatch,
    _fflag_wno_conversion,
    _fflag_wno_maybe_uninitialized,
    _fflag_wno_unused_dummy_argument,
    _fflag_wno_unused_label,
    _fflag_wno_unused_variable,
    _flags_wno_unused_parameter,
    _fflag_wno_tabs,
    _fflag_wno_compare_reals,
    _fflag_wno_character_truncation,
    _fflag_wno_surprising,
)

incdir_numpy = run_command(
    py, '-c',
    'import numpy; print(numpy.get_include())',
    check: true,
).stdout().strip()

inc_np = include_directories(incdir_numpy)

incdir_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

# Share this object across multiple modules.
fortranobject_lib = static_library('_fortranobject',
  fortranobject_c,
  c_args: c_ignore_warnings,
  dependencies: py_dep,
  include_directories: [inc_np, inc_f2py],
)
fortranobject_dep = declare_dependency(
  link_with: fortranobject_lib,
  include_directories: [inc_np, inc_f2py],
)

openmp_dep = dependency('OpenMP')

subdir('smash')