python_sources = [
    '__init__.py',
    '_f90wrap_decorator.py',
]

c_sources = [
    'tapenade/adContext.c',
    'tapenade/adOMP.c',
    'tapenade/adStack.c',
]

f77_sources = [
    'external/lbfgsb.f'
]

m_f90_sources = [
    'common/m_array_creation.f90',
    'common/m_array_manipulation.f90',
    'common/m_screen_display.f90',
]

md_f90_sources = [
    'common/md_constant.f90',
    'common/md_stats.f90',

    'derived_type/md_checkpoint_variable.f90',

    'forward/md_simulation.f90',

    'operator/md_gr_operator.f90',
    'operator/md_routing_operator.f90',
    'operator/md_snow_operator.f90',
    'operator/md_vic3l_operator.f90',

    'optimize/md_regularization.f90',
]

mw_f90_sources = [
    'forward/mw_forward.f90',

    'optimize/mw_optimize.f90',

    'routine/mw_atmos_statistic.f90',
    'routine/mw_interception_capacity.f90',
    'routine/mw_mask.f90',

    'signal_analysis/mw_prcp_indices.f90',
]

mwd_f90_sources = [
    'derived_type/mwd_atmos_data.f90',
    'derived_type/mwd_common_options.f90',
    'derived_type/mwd_control.f90',
    'derived_type/mwd_cost_options.f90',
    'derived_type/mwd_input_data.f90',
    'derived_type/mwd_mesh.f90',
    'derived_type/mwd_optimize_options.f90',
    'derived_type/mwd_options.f90',
    'derived_type/mwd_output.f90',
    'derived_type/mwd_parameters.f90',
    'derived_type/mwd_physio_data.f90',
    'derived_type/mwd_response_data.f90',
    'derived_type/mwd_response.f90',
    'derived_type/mwd_returns.f90',
    'derived_type/mwd_rr_parameters.f90',
    'derived_type/mwd_rr_states.f90',
    'derived_type/mwd_serr_mu_parameters.f90',
    'derived_type/mwd_serr_sigma_parameters.f90',
    'derived_type/mwd_setup.f90',
    'derived_type/mwd_sparse_matrix.f90',
    'derived_type/mwd_u_response_data.f90',

    'external/mwd_bayesian_tools.f90',

    'optimize/mwd_cost.f90',

    'routine/mwd_atmos_manipulation.f90',
    'routine/mwd_parameters_manipulation.f90',
    'routine/mwd_sparse_matrix_manipulation.f90',

    'signal_analysis/mwd_metrics.f90',
    'signal_analysis/mwd_signatures.f90',
]

f90_sources = [
    'forward/forward.f90',
] + m_f90_sources + md_f90_sources + mw_f90_sources + mwd_f90_sources

if host_machine.system() == 'windows'
    f90_sources += 'forward/forward_db.f90'
else
    if get_option('use-openmp')
        f90_sources += 'forward/forward_openmp_db.f90'
    else
        f90_sources += 'forward/forward_db.f90'
    endif
endif

f90wrap = [
    py,
    files('../../f90wrap/generate_f90wrap.py'),
    '@INPUT@',
    '-k', files('../../f90wrap/kind_map'),
    '--build-dir', '@OUTDIR@',
]

f90wrap_f90_output = []
f90wrap_py_output = []
f90wrap_f90_install_dir = []
f90wrap_py_install_dir = []
foreach f: mw_f90_sources + mwd_f90_sources
    basename = f.split('/')[-1]
    f90wrap_f90_output += 'f90wrap_' + basename
    f90wrap_py_output += '_' + basename.replace('f90', 'py')
    f90wrap_f90_install_dir += false
    f90wrap_py_install_dir += py.get_install_dir() / 'smash'/ 'fcore'
endforeach

f90wrap_sources = custom_target(
    input: mw_f90_sources + mwd_f90_sources,
    output: [f90wrap_f90_output, f90wrap_py_output],
    command: f90wrap + ['-m', 'libfcore'],
    install: true,
    install_dir: [f90wrap_f90_install_dir, f90wrap_py_install_dir],
)

f90wrap_f90_sources = []
foreach i : range(f90wrap_f90_output.length())
    f90wrap_f90_sources += f90wrap_sources[i]
endforeach

f2py_f90wrap = ['f2py-f90wrap', '@INPUT@', '--build-dir', '@OUTDIR@', '--lower']

f2py_f90wrap_sources = custom_target(
    input: f90wrap_f90_sources,
    output: ['_libfcoremodule.c', '_libfcore-f2pywrappers.f'],
    command: f2py_f90wrap + ['-m', '_libfcore'],
)

py.install_sources(
    python_sources,
    subdir: 'smash' / 'fcore',
    preserve_path: true,
)

link_args = []
dependencies = [fortranobject_dep]
if host_machine.system() == 'windows'
    link_args += ['-lquadmath', '-static']
else
    if get_option('use-openmp')
        dependencies += openmp_dep
    endif
endif

py.extension_module(
    '_libfcore',
    c_sources, f77_sources, f90_sources, f90wrap_f90_sources, f2py_f90wrap_sources,
    c_args: ['-O3', c_ignore_warnings],
    fortran_args: ['-O3', '-march=native', '-cpp', fortran_ignore_warnings],
    link_args: link_args,
    dependencies: dependencies,
    install: true,
    subdir: 'smash' / 'fcore',
)