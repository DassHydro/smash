#% Meson options (see meson.options)
use-openmp := true

#% Build smash
all:
	pip install -Csetup-args=-Duse-openmp=$(use-openmp) .

#% Build smash in editable mode
edit:
	pip install -Csetup-args=-Duse-openmp=$(use-openmp) --no-build-isolation --config-settings=editable-verbose=true --editable .

TAP_FILES := $(addprefix smash/fcore/*/,mwd_*.f90 md_*.f90 forward.f90)

#% Generate tapenade files (adjoint and tangent linear models)
tap:
	python tapenade/generate_tapenade.py $(TAP_FILES) --build-dir smash/fcore/forward --openmp -m forward_openmp
	python tapenade/generate_tapenade.py $(TAP_FILES) --build-dir smash/fcore/forward -m forward

#% Compare tapenade files
tap-cmp:
	python tapenade/generate_tapenade.py $(TAP_FILES) --build-dir tapenade --openmp -m forward_openmp
	python tapenade/generate_tapenade.py $(TAP_FILES) --build-dir tapenade -m forward
	cmp tapenade/forward_openmp_db.f90 smash/fcore/forward/forward_openmp_db.f90
	cmp tapenade/forward_db.f90 smash/fcore/forward/forward_db.f90

#% Generate sphinx documentation
doc:
	make html -C doc

#% Clean sphinx documentation
doc-clean:
	make clean -C doc

#% Testing code with pytest
test:
	cd smash/tests ; pytest

#% Testing code with pytest and coverage
test-coverage:
	cd smash/tests ; pytest --cov-report term --cov-report html --cov=smash

#% Generate baseline for test with args (see argparser in gen_baseline.py)
test-baseline:
	cd smash/tests ; python3 generate_baseline.py

#% Format Python files with ruff and Fortran files with fprettify
format:
	ruff format
	fprettify -e forward_db.f90 -e forward_openmp_db.f90 -e f90wrap -f .f90 --indent 4 -r smash

#% Check Python files with ruff linters
check:
	ruff check --fix --exit-non-zero-on-fix

#% Clean
clean:
	@rm -rf build

.PHONY: all edit tap tap-cmp doc doc-clean test test-baseline format check clean