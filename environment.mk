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
create-environment: create-environment-$(UNAME_SYS) ## Set up python environment
	@rm -rf $(PYTHON_ENV_STAMP_DIR)

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
create-environment-Darwin:
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

$(PYTHON_ENV_STAMP):
	@rm -rf $(PYTHON_ENV_STAMP_DIR_CURRENT)
	@mkdir -p $(PYTHON_ENV_STAMP_DIR_CURRENT)
	@touch $(PYTHON_ENV_STAMP)

$(PREREQ_STAMP): $(PYTHON_ENV_STAMP)
ifeq ($(UNAME_ARCH), arm64)
ifdef CONDA_ENV_NAME
ifeq ($(CONDA_ENV_NAME), base)
	@echo 'Please create a conda environment and make it active'
	@exit 1
else
	echo "installing conda deps"
	@conda install -y $(call get_python_package_version,tokenizers) \
                         $(call get_python_package_version,xgboost) \
                         $(call get_python_package_version,lightgbm) \
                         $(call get_python_package_version,h5py) \
                         $(call get_python_package_version,pyarrow)
endif
else
	@echo 'Not inside a conda environment or conda not installed, this is a requirement for Apple arm processors'
	@echo 'See https://github.com/conda-forge/miniforge/'
	@exit 1
endif
endif
	@touch $(PREREQ_STAMP)

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
	@find . -type f -name '*.pyc' -not -path "*build/$(PROJECT_NAME)/*" -delete
	@find . -type d -name "__pycache__" -not -path "*build/$(PROJECT_NAME)/*" | xargs rm -rf {};
	@rm -rf $(OUTPUTS_DIR)
	@echo 'done.'
