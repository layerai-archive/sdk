ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
INSTALL_STAMP := .install.stamp
POETRY := $(shell command -v poetry 2> /dev/null)
IN_VENV := $(shell echo $(CONDA_DEFAULT_ENV)$(CONDA_PREFIX)$(VIRTUAL_ENV))

.DEFAULT_GOAL:=help

install: check-poetry apple-arm-prereq-install $(INSTALL_STAMP) ## Install dependencies
$(INSTALL_STAMP): pyproject.toml poetry.lock
ifdef IN_VENV
	$(POETRY) install
else
	$(POETRY) install --remove-untracked
endif
	touch $(INSTALL_STAMP)

.PHONY: check-poetry
check-poetry:
	@if [ -z $(POETRY) ]; then echo "Poetry could not be found. See https://python-poetry.org/docs/"; exit 2; fi

.PHONY: check-colab-violations
check-colab-violations: $(INSTALL_STAMP) ## Check that colab pre-installed packages are not clashing with ours
	$(POETRY) run python validate_colab_dependencies.py

.PHONY: clean
clean: ## Resets development environment.
	@echo 'cleaning repo...'
	@rm -rf .mypy_cache
	@rm -rf .pytest_cache
	@rm -f .coverage
	@find . -type d -name '*.egg-info' | xargs rm -rf {};
	@find . -depth -type d -name '*.egg-info' -delete
	@rm -rf dist/
	@rm -f $(INSTALL_STAMP)
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name "__pycache__" | xargs rm -rf {};
	@echo 'done.'

.PHONY: deepclean
deepclean: clean ## Resets development environment including test credentials and venv
	@rm -rf `poetry env info -p`
	@rm -f $(TEST_TOKEN_FILE)

.PHONY: help
help: ## Show this help message.
	@echo 'usage: make [target]'
	@echo
	@echo 'targets:'
	@grep -E '^[8+a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo
