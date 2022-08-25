ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
E2E_TEST_HOME := $(ROOT_DIR)/build/e2e-home
E2E_TEST_SELECTOR := test/e2e
E2E_TEST_PARALLELISM := 16
TEST_TOKEN_FILE := .test-token
POETRY := $(shell command -v poetry 2> /dev/null)
IN_VENV := $(shell echo $(CONDA_DEFAULT_ENV)$(CONDA_PREFIX)$(VIRTUAL_ENV))
CONDA_ENV_NAME := $(shell echo $(CONDA_DEFAULT_ENV))
UNAME_SYS := $(shell uname -s)
UNAME_ARCH := $(shell uname -m)
REQUIRED_POETRY_VERSION := 1.1.15
PROJECT_NAME := sdk
COLAB_TEST_HOME := $(ROOT_DIR)/build/colab-test
DOCKER_IMAGE_NAME = layerco/colab-lite
OUTPUTS_DIR := $(ROOT_DIR)/build/outputs
PYTHON_ENV_STAMP_DIR := $(OUTPUTS_DIR)/python-env-stamps
PYTHON_ENV_STAMP_PYTHON_PATH := $(shell which python | tr -d '\n')
PYTHON_VERSION := $(shell python --version | tr -d '\n' | sed 's/ //g')
PYTHON_ENV_STAMP_DIR_CURRENT := $(PYTHON_ENV_STAMP_DIR)$(PYTHON_ENV_STAMP_PYTHON_PATH)/$(PYTHON_VERSION)
PYTHON_ENV_STAMP := $(PYTHON_ENV_STAMP_DIR_CURRENT)/.python.stamp
PREREQ_STAMP := $(PYTHON_ENV_STAMP_DIR_CURRENT)/.prereq.stamp
INSTALL_STAMP := $(PYTHON_ENV_STAMP_DIR_CURRENT)/.install.stamp


ifneq ($(shell $(POETRY) --version | awk '{print $$3}'), $(REQUIRED_POETRY_VERSION))
$(error "Please use Poetry version $(REQUIRED_POETRY_VERSION). Simply run: poetry self update $(REQUIRED_POETRY_VERSION)")
endif
