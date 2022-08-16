from unittest.mock import patch

import layer


@patch.object(layer, "log")
@patch.object(layer, "init")
@patch.object(layer, "login_with_api_key")
def test_pl_logger(log, init, login_with_api_key):
    from layer.logged_data.loggers.pytorch_lightning import PytorchLightningLogger

    project_name = "example_project"
    api_key = "example_api_key"
    logger = PytorchLightningLogger(project_name=project_name, api_key=api_key)

    # Check logging simple text
    layer.log.reset_mock()
    logger.log_text(key="text", text="value")
    layer.log.assert_called_once_with({"text": "value"})

    # Check logging a dataframe
    columns = ["col1"]
    rows = [["val1"]]
    layer.log.reset_mock()
    import pandas as pd

    df = pd.DataFrame(columns=columns, data=rows)
    logger.log_table(key="dataframe", dataframe=df)
    args, kwargs = layer.log.call_args
    assert df.equals(args[0]["dataframe"])

    # Check logging a table
    layer.log.reset_mock()
    logger.log_table(key="table", columns=columns, data=rows)
    args, kwargs = layer.log.call_args
    assert df.equals(args[0]["table"])

    # Check logging an image
    layer.log.reset_mock()
    img = "1.jpg"
    logger.log_image(key="image", image=img)
    args, kwargs = layer.log.call_args
    assert img == args[0]["image"].img

    # Check logging stepped image
    step = 10
    layer.log.reset_mock()
    logger.log_image(key="image", image=img, step=step)
    args, kwargs = layer.log.call_args
    assert img == args[0]["image"].img
    assert step == kwargs["step"]

    # Check logging a video
    layer.log.reset_mock()
    video = "test.mp4"
    logger.log_video(key="video", video=video)
    layer.log.assert_called_once_with({"video": video}, step=None)

    # Check logging torch tensors as video
    import torch

    fps = 24
    layer.log.reset_mock()
    video_tensor = torch.rand(10, 3, 100, 200)
    logger.log_video(key="video", video=video_tensor, fps=fps)
    args, kwargs = layer.log.call_args
    logged_video = args[0]["video"]
    assert video_tensor.equal(logged_video.video)
    assert fps == logged_video.fps
