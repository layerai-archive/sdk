# We need -it to be able to interrupt locally, however on CI -t leads to the the input device is not a TTY error
ifdef CI
	DOCKER_RUN = docker run -v $(shell pwd):/usr/src/app:ro -v $(shell pwd)/dist:/usr/src/app/dist:rw --rm --platform=linux/amd64 -e LAYER_API_KEY=$(shell cat .test-token) --name colab-test $(DOCKER_IMAGE_NAME)
else
	DOCKER_RUN = docker run -v $(shell pwd):/usr/src/app:ro -v $(shell pwd)/dist:/usr/src/app/dist:rw --rm --platform=linux/amd64 -e LAYER_API_KEY=$(shell cat .test-token) --name colab-test -it $(DOCKER_IMAGE_NAME)
endif


.PHONY: colab-test-internal
colab-test-internal: $(TEST_TOKEN_FILE) $(COLAB_TEST_HOME)/test_import_login_init.ipynb colab-test-pull dist/layer-0.10.0-py3-none-any.whl
	@$(DOCKER_RUN)

.PHONY: colab-test-internal-local
colab-test-internal-local: $(TEST_TOKEN_FILE) $(COLAB_TEST_HOME)/test_import_login_init.ipynb dist/layer-0.10.0-py3-none-any.whl
	@$(DOCKER_RUN)

dist/layer-0.10.0-py3-none-any.whl:
	@poetry build --format wheel

.PHONY: colab-test-pull
colab-test-pull:
	@docker pull $(DOCKER_IMAGE_NAME)

.DELETE_ON_ERROR: $(COLAB_TEST_HOME)/test_import_login_init.ipynb

$(COLAB_TEST_HOME)/test_import_login_init.ipynb: test/colab/test_import_login_init.ipynb
	@mkdir -p $(COLAB_TEST_HOME)
# VSC saves notebooks with `null` in place of `0` for `execution_count`, e.g. `"execution_count": null,`.\
`nbconvert` crashes on these, so we fix that here. 
	@cp test/colab/test_import_login_init.ipynb $(COLAB_TEST_HOME)/test_import_login_init.ipynb
	@sed -i'' -e "s/\"execution_count\": null,/\"execution_count\": 0,/g" $(COLAB_TEST_HOME)/test_import_login_init.ipynb
