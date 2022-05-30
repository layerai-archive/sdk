<!---
Copyright 2022 Layer. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
    <br>
    <a href="https://layer.ai">
          <img src="https://app.layer.ai/assets/layer_wordmark_black.png" width="40%"
alt="Layer"/>
    </a>
    <br>
<p>
<p align="center">
    <a href="https://github.com/layerai/sdk/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/layerai/sdk.svg?color=blue">
    </a>
    <a href="https://docs.app.layer.ai">
        <img alt="Documentation" src="https://img.shields.io/badge/docs-online-success">
    </a>
    <a href="https://github.com/layerai/sdk/actions/workflows/release.yml">
        <img alt="Build" src="https://img.shields.io/github/workflow/status/layerai/sdk/Release">
    </a>
    <a href="https://pypi.python.org/pypi/layer">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/layer.svg">
    </a>
    <a href="https://github.com/layerai/.github/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/contributor%20covenant-v2.1%20adopted-blueviolet.svg">
    </a>
</p>

## Layer - Metadata Store for Production ML

![Layer - Metadata store for production ML](https://app.layer.ai/assets/layer_metadata_store.png)


[Layer](https://layer.ai) helps you build, train and track all your machine learning project metadata including ML models and datasets with semantic versioning, extensive artifact logging and dynamic reporting with local↔cloud training

**[Start for Free now!](https://app.layer.ai/login?returnTo=%2Fgetting-started)**

## Getting Started

Install Layer:
```shell
pip install layer --upgrade
```

Login to your free account and initialize your project:
```python
import layer
layer.login()
layer.init("my-first-project")
```

Decorate your training function to register your model to Layer:
```python
from layer.decorators import model

@model("my-model")
def train():
    from sklearn import datasets
    from sklearn.svm import SVC
    iris = datasets.load_iris()
    clf = SVC()
    clf.fit(iris.data, iris.target)
    return clf

train()
```

Now you can fetch your model from Layer:

```python
import layer

clf = layer.get_model("my-model:1.1").get_train()
clf

# > SVC()
```

[**🚀 Try in Google Colab now!**](https://colab.research.google.com/github/layerai/examples/blob/main/tutorials/add-models-to-layer/how_to_add_models_to_layer.ipynb)

## Reporting bugs
You have a bug, a request or a feature? Let us know on [Slack](https://bit.ly/layercommunityslack) or [open an issue](https://github.com/layerai/sdk/issues/new/choose)

## Contributing code
Do you want to help us build the best metadata store? Check out the [Contributing Guide](https://github.com/layerai/sdk/blob/main/CONTRIBUTING.md)

## Learn more
- Join our [Slack Community ](https://bit.ly/layercommunityslack) to connect with other Layer users
- Visit the [examples repo](https://github.com/layerai/examples) for more inspiration
- Browse [Community Projects](https://layer.ai/community) to see more use cases
- Check out the [Documentation](https://docs.app.layer.ai)
- [Contact us](https://layer.ai/contact-us) for your questions
