# Makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option. $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@mkdir $(BUILDDIR)_tmp 
	@cp -r $(BUILDDIR)/* $(BUILDDIR)_tmp 2>/dev/null || true
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) --fail-on-warning || \
    (echo "An error occurred while generating the documentation. Returning to previous version." && \
	rm -rf $(BUILDDIR) && mv $(BUILDDIR)_tmp $(BUILDDIR) && false)
	@rm -rf $(BUILDDIR)_tmp
	
clean:
	@rm -rf $(BUILDDIR)
	@rm -rf $(SOURCEDIR)/savefig
	@rm -rf *.hdf5 *yaml
	@find $(SOURCEDIR)/api_reference/ -type d -name "smash" -exec rm -rf {} +
