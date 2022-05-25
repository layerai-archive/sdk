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
          <img src=".github/assets/layer_logo_light.png#gh-light-mode-only" width="512" alt="Layer"/>
          <img src=".github/assets/layer_logo_light.png#gh-dark-mode-only" width="512" alt="Layer"/>
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
    <a href="https://github.com/layerai/sdk/actions/workflows/check.yml">
        <img alt="Build" src="https://img.shields.io/github/workflow/status/layerai/sdk/Check">
    </a>
    <a href="https://pypi.python.org/pypi/layer">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/layer.svg">
    </a>
    <a href="https://github.com/layer/sdk/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/contributor%20covenant-v2.1%20adopted-blueviolet.svg">
    </a>
</p>

[Layer](https://layer.ai) helps you build, train and track all your machine learning project metadata including ML models and datasets with semantic versioning, extensive artifact logging and dynamic reporting with local↔cloud training

[Start for Free now!](https://app.layer.ai/login?returnTo=%2Fgetting-started)

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

And decorate your training function to register your model to Layer:
```python
from layer.decorators import model

@model("my-model")
def train():
    model = ...
    model.fit(..)
    return model

train()
```

### [Try in a colab now!](https://docs.app.layer.ai/docs/getting-started)

## Reporting bugs 
You have a bug, a request or a feature? Let us know on [Slack](https://bit.ly/layercommunityslack) or [open an issue](https://github.com/layerai/sdk/issues/new/choose)

## Contributing code
Do you want to help us build the best metadata store? Check out the [Contributing Guide](./CONTRIBUTING.md)

## Learn more
- Join our [Slack Community ](https://bit.ly/layercommunityslack) to connect with other Layer users
- Visit [Layer Examples Repo](https://github.com/layerai/examples) for more examples
- Browse [Community Projects](https://layer.ai/community) to see more use cases
- Check out [Layer Documentation](https://docs.layer.ai)
- [Contact us](https://layer.ai/contact-us?interest=notebook) for your questions

