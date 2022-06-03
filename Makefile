ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
INSTALL_STAMP := .install.stamp
E2E_TEST_HOME := $(ROOT_DIR)/build/e2e-home
TEST_TOKEN_FILE := .test-token
POETRY := $(shell command -v poetry 2> /dev/null)
IN_VENV := $(shell echo $(CONDA_DEFAULT_ENV)$(CONDA_PREFIX)$(VIRTUAL_ENV))

.DEFAULT_GOAL:=help

install: $(INSTALL_STAMP) ## Install dependencies
$(INSTALL_STAMP): pyproject.toml poetry.lock
	@if [ -z $(POETRY) ]; then echo "Poetry could not be found. See https://python-poetry.org/docs/"; exit 2; fi
ifdef IN_VENV
	$(POETRY) install
else
	$(POETRY) install --remove-untracked
endif
	touch $(INSTALL_STAMP)

.PHONY: test
test: $(INSTALL_STAMP) ## Run unit tests
	$(POETRY) run pytest test/unit --cov .

test-login: $(TEST_TOKEN_FILE)
$(TEST_TOKEN_FILE):
	@stty -echo
	@printf "Enter test user token for https://app.layer.ai: "
	@read TOKEN && echo $$TOKEN > $(TEST_TOKEN_FILE)
	@stty echo

.PHONY: e2e-test
e2e-test: $(INSTALL_STAMP) $(TEST_TOKEN_FILE)
	@rm -rf $(E2E_TEST_HOME)
ifdef CI
	$(eval DATADOG_ARGS := --ddtrace-patch-all --ddtrace)
endif
	LAYER_DEFAULT_PATH=$(E2E_TEST_HOME) $(POETRY) run python build_scripts/sdk_login.py $(TEST_TOKEN_FILE)
	LAYER_DEFAULT_PATH=$(E2E_TEST_HOME) $(POETRY) run pytest test/e2e -s -n 16 -vv $(DATADOG_ARGS)

.PHONY: format
format: $(INSTALL_STAMP) ## Apply formatters
	$(POETRY) run isort --profile=black --lines-after-imports=2 .
	$(POETRY) run black .

.PHONY: lint
lint: $(INSTALL_STAMP) ## Run all linters
	$(POETRY) run isort --check-only .
	$(POETRY) run black --check . --diff
	$(POETRY) run flake8 .
	$(POETRY) run pylint  --recursive yes .
	$(POETRY) run mypy .
	$(POETRY) run bandit -c pyproject.toml -r .

.PHONY: check-package-loads
check-package-loads: ## Check that we can load the package without the dev dependencies
	@rm -f $(INSTALL_STAMP)
ifdef IN_VENV
	$(POETRY) install --no-dev
else
	$(POETRY) install --no-dev --remove-untracked
endif
	$(POETRY) run python -c "import layer"

.PHONY: check
check: test lint ## Run test and lint

.PHONY: publish
publish: ## Publish to PyPi - should only run in CI
	@test $${PATCH_VERSION?PATCH_VERSION expected}
	@test $${PYPI_USER?PYPI_USER expected}
	@test $${PYPI_PASSWORD?PYPI_PASSWORD expected}
	$(eval CURRENT_VERSION := $(shell $(POETRY) version --short))
	$(eval PARTIAL_VERSION=$(shell echo $(CURRENT_VERSION) | grep -Po '.*(?=\.)'))
	$(POETRY) version $(PARTIAL_VERSION).$(PATCH_VERSION)
	$(POETRY) publish --build --username $(PYPI_USER) --password $(PYPI_PASSWORD)
	$(POETRY) version $(CURRENT_VERSION)

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
