#
# Environment Management Makefile

define conda_environment_file
name: sdk\\n
channels:\\n
  - default\\n
  - apple\\n
  - conda-forge\\n
dependencies:\\n
  - python=$(shell cat .python-version)\\n
  - pip\\n
  - poetry=$(REQUIRED_POETRY_VERSION)
endef

.PHONY: create-environment
create-environment: create-environment-$(UNAME_SYS)

.PHONY: create-environment-Linux
create-environment-Linux: .python-version
ifeq ($(shell which pyenv),)
	@echo "pyenv is not Installed, please install before creating an environment"
	@exit 1
else
	@pyenv install -s $(shell cat $<)
	@echo "pyenv Python version $(shell cat $<) installed."
	@echo "Activate shell with poetry shell"
endif

.PHONY: create-environment-Darwin
create-environment-Darwin: ## Set up virtual (conda) environment for MacOS
ifeq ($(UNAME_ARCH), arm64)
ifdef CONDA_ENV_NAME
	$(shell echo $(conda_environment_file) > .environment.M1.yml)
	$(CONDA_EXE) env update -p build/$(PROJECT_NAME) -f .environment.M1.yml
	@echo
	@echo "Conda env is available and can be activated."
else
	$(error Unsupported Environment. Please use conda)
endif
endif

.PHONY: delete-environment
delete-environment: delete-environment-$(UNAME_SYS)

.PHONY: delete-environment-Linux
delete-environment-Linux:
	@echo "No action needed"

.PHONY: delete-environment-Darwin ## Delete the virtual (conda) environment
delete-environment-Darwin:
ifeq ($(UNAME_ARCH), arm64)
	@echo "Deleting conda environment."
	rm -fr build/$(PROJECT_NAME)
endif

.PHONY: jupyter
jupyter: install ## Start a jupyter notebook with editable layer package
	@IPYTHONDIR=$(ROOT_DIR)/build/ipython poetry run jupyter-notebook

.PHONY: clean
clean: ## Resets development environment.
	@echo 'cleaning repo...'
	@rm -rf .mypy_cache
	@rm -rf .pytest_cache
	@rm -f .coverage
	@find . -type d -name '*.egg-info' -not -path "*build/$(PROJECT_NAME)/*" | xargs rm -rf {};
	@find . -depth -type d -name '*.egg-info' -not -path "*build/$(PROJECT_NAME)/*" -delete
	@rm -rf dist/
	@rm -f $(INSTALL_STAMP)
	@rm -f $(COLAB_IMAGE_BUILD_STAMP)
	@rm -rf $(ROOT_DIR)/build/
	@find . -type f -name '*.pyc' -not -path "*build/$(PROJECT_NAME)/*" -delete
	@find . -type d -name "__pycache__" -not -path "*build/$(PROJECT_NAME)/*" | xargs rm -rf {};
	@echo 'done.'
