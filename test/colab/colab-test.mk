.PHONY: colab-test-internal
colab-test-internal: $(TEST_TOKEN_FILE) $(COLAB_TEST_HOME)/test_import_login_init.ipynb colab-test-pull
	$(DOCKER_RUN)

.PHONY: colab-test-internal-local
colab-test-internal-local: $(TEST_TOKEN_FILE) $(COLAB_TEST_HOME)/test_import_login_init.ipynb colab-test-build
	$(DOCKER_RUN)

.PHONY: colab-test-pull
colab-test-pull:
	@docker pull $(DOCKER_IMAGE_NAME)

.PHONY: colab-test-build
colab-test-build: $(COLAB_IMAGE_BUILD_STAMP)
$(COLAB_IMAGE_BUILD_STAMP): $(COLAB_TEST_HOME)/requirements-fixed.txt test/colab/Dockerfile
	@docker build -t $(DOCKER_IMAGE_NAME) -f test/colab/Dockerfile .
	@touch $(COLAB_IMAGE_BUILD_STAMP)

.DELETE_ON_ERROR: $(COLAB_TEST_HOME)/requirements-fixed.txt $(COLAB_TEST_HOME)/test_import_login_init.ipynb

$(COLAB_TEST_HOME)/requirements-fixed.txt: test/colab/requirements.txt
	@mkdir -p $(COLAB_TEST_HOME)
# Fix requirements.txt to make them compatible with running on M1 mac/linux, outside of colab itself.
	@./test/colab/fix-requirements.sh

$(COLAB_TEST_HOME)/test_import_login_init.ipynb: test/colab/test_import_login_init.ipynb
	mkdir -p $(COLAB_TEST_HOME)
# VSC saves notebooks with `null` in place of `0` for `execution_count`, e.g. `"execution_count": null,`.\
`nbconvert` crashes on these, so we fix that here. 
	pwd
	ls -al
	ls -al $(COLAB_TEST_HOME)
	cp test/colab/test_import_login_init.ipynb $(COLAB_TEST_HOME)/test_import_login_init.ipynb
	ls -al $(COLAB_TEST_HOME)
	sed -i '' "s/\"execution_count\": null,/\"execution_count\": 0,/g" $(COLAB_TEST_HOME)/test_import_login_init.ipynb
