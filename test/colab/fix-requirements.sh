cp test/colab/requirements.txt build/colab-test/requirements-fixed.txt

# Ensure setuptools version lower than 58 is used, as 58 and above break the install due to use_2to3 not being available
sed -i'' -e "1s/^/setuptools<58\n/" build/colab-test/requirements-fixed.txt

# Remove Tensorflow libraries due to AVX missing on M1 mac
sed -i'' -e "s/kapre==/# kapre==/" build/colab-test/requirements-fixed.txt
sed -i'' -e "s/keras==/# keras==/" build/colab-test/requirements-fixed.txt
sed -i'' -e "s/tensorflow==/# tensorflow==/" build/colab-test/requirements-fixed.txt

# Remove non-existing versions
sed -i'' -e "s|google-colab @ file:///colabtools/dist/google-colab-1.0.0.tar.gz|git+https://github.com/googlecolab/colabtools.git@master#egg=google-colab|" build/colab-test/requirements-fixed.txt
sed -i'' -e "s/python-apt==0.0.0/# python-apt==0.0.0/" build/colab-test/requirements-fixed.txt
sed -i'' -e "s/screen-resolution-extra==0.0.0/# screen-resolution-extra==0.0.0/" build/colab-test/requirements-fixed.txt
sed -i'' -e "s/xkit==0.0.0/# xkit==0.0.0/" build/colab-test/requirements-fixed.txt

# Upgrade non-available versions
sed -i'' -e "s/cvxpy==1.0.31/cvxpy==1.1.0/" build/colab-test/requirements-fixed.txt
sed -i'' -e "s/pygobject==3.26.1/pygobject==3.27.0/" build/colab-test/requirements-fixed.txt