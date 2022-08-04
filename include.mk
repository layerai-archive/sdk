ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
INSTALL_STAMP := .install.stamp
E2E_TEST_HOME := $(ROOT_DIR)/build/e2e-home
E2E_TEST_SELECTOR := test/e2e
E2E_TEST_PARALLELISM := 16
TEST_TOKEN_FILE := .test-token
POETRY := $(shell command -v poetry 2> /dev/null)
IN_VENV := $(shell echo $(CONDA_DEFAULT_ENV)$(CONDA_PREFIX)$(VIRTUAL_ENV))
CONDA_ENV_NAME := $(shell echo $(CONDA_DEFAULT_ENV))
UNAME_SYS := $(shell uname -s)
UNAME_ARCH := $(shell uname -m)
REQUIRED_POETRY_VERSION := 1.1.14
PROJECT_NAME := sdk
COLAB_TEST_HOME := $(ROOT_DIR)/build/colab-test
COLAB_IMAGE_BUILD_STAMP := .image-built.stamp
DOCKER_IMAGE_NAME = layerco/colab-lite

