python_sources = [
    '__init__.py',
    '_constant.py',
    '_version.py',
]

py.install_sources(
    python_sources,
    subdir: 'smash',
    preserve_path: true,
)

python_subdirs = ['core', 'factory', 'io', 'util']

foreach subdir: python_subdirs
    install_subdir(subdir, install_dir: py.get_install_dir() / 'smash')
endforeach

# Manage Fortran wrapping
subdir('factory/mesh')
subdir('fcore')