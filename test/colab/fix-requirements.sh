cp test/colab/requirements.txt build/colab-test/requirements-fixed.txt

# Ensure setuptools version lower than 58 is used, as 58 and above break the install due to use_2to3 not being available
sed -i '' "1s/^/setuptools<58\n/" build/colab-test/requirements-fixed.txt

# Remove Tensorflow libraries due to AVX missing on M1 mac
sed -i '' "s/kapre==/# kapre==/" build/colab-test/requirements-fixed.txt
sed -i '' "s/keras==/# keras==/" build/colab-test/requirements-fixed.txt
sed -i '' "s/tensorflow==/# tensorflow==/" build/colab-test/requirements-fixed.txt

# Remove non-existing versions
sed -i '' "s/google-colab @/# google-colab @/" build/colab-test/requirements-fixed.txt
sed -i '' "s/python-apt==0.0.0/# python-apt==0.0.0/" build/colab-test/requirements-fixed.txt
sed -i '' "s/screen-resolution-extra==0.0.0/# screen-resolution-extra==0.0.0/" build/colab-test/requirements-fixed.txt
sed -i '' "s/xkit==0.0.0/# xkit==0.0.0/" build/colab-test/requirements-fixed.txt

# Upgrade non-available versions
sed -i '' "s/cvxpy==1.0.31/cvxpy==1.1.0/" build/colab-test/requirements-fixed.txt
sed -i '' "s/pygobject==3.26.1/pygobject==3.27.0/" build/colab-test/requirements-fixed.txt