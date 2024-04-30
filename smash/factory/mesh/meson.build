f2py = [py, '-m', 'numpy.f2py', '@INPUT@', '--build-dir', '@OUTDIR@', '--lower']

f2py_sources = custom_target(
    input: 'mw_mesh.f90',
    output: ['_libmeshmodule.c', '_libmesh-f2pywrappers2.f90'],
    command: f2py + ['-m', '_libmesh']
)

py.extension_module(
    '_libmesh',
    'mw_mesh.f90', f2py_sources,
    c_args: c_ignore_warnings,
    fortran_args: fortran_ignore_warnings,
    dependencies: fortranobject_dep,
    install: true,
    subdir: 'smash' / 'factory' / 'mesh',
)