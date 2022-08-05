#!/bin/bash
set -eo pipefail

trap 'wait;' SIGINT SIGTERM

jupyter nbconvert --to notebook --execute build/colab-test/test_import_login_init.ipynb --ExecutePreprocessor.timeout=300 --stdout --debug

wait