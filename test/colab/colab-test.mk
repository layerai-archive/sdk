.PHONY: colab-test-internal
colab-test-internal: $(TEST_TOKEN_FILE) colab-test-pull build/colab-test/test_import_login_init.ipynb
	$(DOCKER_RUN)

.PHONY: colab-test-internal-local
colab-test-internal-local: $(TEST_TOKEN_FILE) colab-test-build build/colab-test/test_import_login_init.ipynb
	$(DOCKER_RUN)

.PHONY: colab-test-pull
colab-test-pull:
	@docker pull $(DOCKER_IMAGE_NAME)

.PHONY: colab-test-build
colab-test-build: $(COLAB_IMAGE_BUILD_STAMP)
$(COLAB_IMAGE_BUILD_STAMP): build/colab-test/requirements-fixed.txt test/colab/Dockerfile
	@docker build -t $(DOCKER_IMAGE_NAME) -f test/colab/Dockerfile .
	@touch $(COLAB_IMAGE_BUILD_STAMP)

.DELETE_ON_ERROR: build/colab-test/requirements-fixed.txt build/colab-test/test_import_login_init.ipynb

build/colab-test/requirements-fixed.txt: test/colab/requirements.txt
	@mkdir -p $(COLAB_TEST_HOME)
# Fix requirements.txt to make them compatible with running on M1 mac/linux, outside of colab itself.
	@./test/colab/fix-requirements.sh

build/colab-test/test_import_login_init.ipynb: test/colab/test_import_login_init.ipynb
	@mkdir -p $(COLAB_TEST_HOME)
# VSC saves notebooks with `null` in place of `0` for `execution_count`, e.g. `"execution_count": null,`.\
`nbconvert` crashes on these, so we fix that here. 
	@cp test/colab/test_import_login_init.ipynb build/colab-test/test_import_login_init.ipynb
	@sed -i '' "s/\"execution_count\": null,/\"execution_count\": 0,/g" build/colab-test/test_import_login_init.ipynb
