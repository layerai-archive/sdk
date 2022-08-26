import warnings


warnings.filterwarnings(
    "ignore",
    "`should_run_async` will not call `transform_cell` automatically in the future.",
    DeprecationWarning,
)

warnings.filterwarnings(
    "ignore",
    "Setuptools is replacing distutils.",
    UserWarning,
)
