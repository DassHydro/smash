#% Build smash
all:
	pip install .

#% Build smash in editable mode
edit:
	pip install --no-build-isolation --config-settings=editable-verbose=true --editable .

#% Generate tapenade file (adjoint and tangent linear models)
tap:
	make -C tapenade

#% Compare tapenade file
tap-cmp:
	make tap-cmp -C tapenade

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
	cd smash/tests ; python3 gen_baseline.py

#% Format Python files with ruff and Fortran files with fprettify
format:
	ruff format
	fprettify -e forward_db.f90 -e f90wrap -f .f90 --indent 4 -r smash

#% Check Python files with ruff linters
check:
	ruff check --fix --exit-non-zero-on-fix

#% Clean
clean:
	@rm -rf build

.PHONY: all edit tap tap-cmp doc doc-clean test test-baseline format check clean