from pathlib import Path
from traceback import FrameSummary
from typing import Any, List, Optional, Set, Type

from yarl import URL

from layer.api.ids_pb2 import RunId

from .status_report import ExecutionStatusReport


RICH_ERROR_COLOR = "rgb(251,147,60)"


class RuntimeMemoryException(Exception):
    def __init__(self, err_details: str):
        super().__init__(f"Memory error: {err_details}")


class FunctionParameterException(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class SparkRuntimeException(Exception):
    pass


class ConfigError(Exception):
    pass


class MissingConfigurationError(ConfigError):
    def __init__(self, path: Path):
        self._path: Path = path
        super().__init__(f"Missing configuration file {path.resolve()}")

    @property
    def path(self) -> Path:
        return self._path


class InvalidConfigurationError(ConfigError):
    def __init__(self, path: Path):
        self._path: Path = path
        super().__init__(f"Invalid configuration file {path.resolve()}")

    @property
    def path(self) -> Path:
        return self._path


class AuthException(Exception):
    pass


class LayerClientException(Exception):
    pass


class LayerClientTimeoutException(LayerClientException):
    pass


class LayerClientResourceNotFoundException(LayerClientException):
    pass


class LayerClientResourceAlreadyExistsException(LayerClientException):
    pass


class LayerResourceExhaustedException(LayerClientException):
    pass


class LayerClientServiceUnavailableException(LayerClientException):
    pass


class ProjectException(Exception):
    pass


class ProjectBaseException(Exception):
    def __init__(self, error_msg: str = "", suggestion: str = ""):
        self._error_msg = error_msg
        self._suggestion = suggestion
        super().__init__(self._format_message())

    @property
    def error_msg(self) -> str:
        return self._error_msg

    @property
    def error_msg_rich(self) -> str:
        return self.error_msg

    @property
    def suggestion(self) -> str:
        return self._suggestion

    @property
    def suggestion_rich(self) -> str:
        return self.suggestion

    def _format_message(self) -> str:
        return f"Error: {self._error_msg},\nSuggestion: {self._suggestion}"


class ProjectExecutionException(ProjectBaseException):
    def __init__(self, run_id: RunId, err_msg: str, suggestion: str):
        super().__init__(err_msg, suggestion)
        self._run_id = run_id

    def run_id(self) -> RunId:
        return self._run_id


class ProjectModelExecutionException(ProjectExecutionException):
    def __init__(self, run_id: RunId, train_id: str, report: ExecutionStatusReport):
        self._report = report
        self._train_id = train_id
        super().__init__(
            run_id,
            self.__format_message(),
            "Add `debug=True` parameter to your `layer.run()` to see server logs",
        )

    @property
    def message(self) -> str:
        return self._report.message

    @property
    def error_msg_rich(self) -> str:
        return (
            f"Train [bold {RICH_ERROR_COLOR}]{self._train_id}[/bold {RICH_ERROR_COLOR}] failed with "
            + f'"{self._report.message}"'
        )

    @property
    def suggestion_rich(self) -> str:
        return self.suggestion

    def __format_message(self) -> str:
        return f"Train {self._train_id} failed with " + f'"{self._report.message}"'

    @staticmethod
    def __format_frame(frame: FrameSummary) -> str:
        return f'File "{frame.filename}" on line {frame.lineno}: "{frame.line}"'


class ProjectHPTExecutionException(ProjectExecutionException):
    def __init__(self, run_id: RunId, tuning_id: str, report: ExecutionStatusReport):
        self._tuning_id = tuning_id
        self._report = report
        super().__init__(
            run_id,
            self.__format_message(),
            "Add `debug=True` parameter to your `layer.run()` to see server logs",
        )

    @property
    def message(self) -> str:
        return self._report.message

    @property
    def error_msg_rich(self) -> str:
        return (
            f"HPT [bold {RICH_ERROR_COLOR}]{self._tuning_id}[/bold {RICH_ERROR_COLOR}] failed with "
            + f'"{self._report.message}"'
        )

    @property
    def suggestion_rich(self) -> str:
        return self.suggestion

    def __format_message(self) -> str:
        return f"HPT {self._tuning_id} failed with " + f'"{self._report.message}"'

    @staticmethod
    def __format_frame(frame: FrameSummary) -> str:
        return f'File "{frame.filename}" on line {frame.lineno}: "{frame.line}"'


class ProjectDatasetBuildExecutionException(ProjectExecutionException):
    def __init__(self, run_id: RunId, dataset_id: str, report: ExecutionStatusReport):
        self._dataset_id = dataset_id
        self._report = report
        super().__init__(
            run_id,
            self.__format_message(),
            "Add `debug=True` parameter to your `layer.run()` to see server logs",
        )

    @property
    def message(self) -> str:
        return self._report.message

    @property
    def error_msg_rich(self) -> str:
        return (
            f'Dataset "[bold {RICH_ERROR_COLOR}]{self._dataset_id}[/bold {RICH_ERROR_COLOR}]" build failed with '
            + f'"{self._report.message}"'
        )

    @property
    def suggestion_rich(self) -> str:
        return self.suggestion

    def __format_message(self) -> str:
        return (
            f'Dataset "{self._dataset_id}" build failed with '
            + f'"{self._report.message}"'
        )

    @staticmethod
    def __format_frame(frame: FrameSummary) -> str:
        return f'File "{frame.filename}" on line {frame.lineno}: "{frame.line}"'


class ProjectInitializationException(ProjectBaseException):
    pass


class ProjectDependencyNotFoundException(ProjectInitializationException):
    pass


class LayerServiceUnavailableExceptionDuringInitialization(
    ProjectInitializationException
):
    def __init__(self, message: str):
        super().__init__(
            f"Layer service unavailable. {message}",
            "Check your internet connection.",
        )


class LayerServiceUnavailableExceptionDuringExecution(ProjectExecutionException):
    def __init__(self, run_id: RunId, message: str):
        super().__init__(
            run_id,
            f"Layer service unavailable. {message}",
            "Check your internet connection.",
        )


class UserNotLoggedInException(ProjectInitializationException):
    def __init__(self) -> None:
        super().__init__("User not logged in.", "Run `layer.login()`")


class UserConfigurationError(ProjectInitializationException):
    def __init__(self, path: Path) -> None:
        super().__init__(
            f"User configuration error in {path.resolve()}",
            "Please make sure that you have the latest version and try to login again.",
        )


class UserWithoutAccountError(ProjectInitializationException):
    def __init__(self, url: URL) -> None:
        super().__init__(
            "You don't have a Layer account yet",
            f"You can create your account and choose a username at {url}.",
        )


class UserAccessTokenExpiredError(ProjectInitializationException):
    def __init__(
        self,
    ) -> None:
        super().__init__("Your session has expired.", "Please, try to login again.")


class ProjectRunnerError(Exception):
    def __init__(self, message: str = "", run_id: Optional[RunId] = None):
        super().__init__(message)
        self._run_id = run_id

    def run_id(self) -> Optional[RunId]:
        return self._run_id


class FunctionDefinitionException(ProjectInitializationException):
    def __init__(
        self,
        param_name: str,
        func_name: str,
        script_path: Path,
        valid_entities: Set[str],
        package_name: str,
    ) -> None:
        msg = f'Parameter "{param_name}" of function "{func_name}" in "{script_path.resolve()}" is not annotated with one of {valid_entities} {package_name} package entities.'
        super().__init__(
            msg, "Please, look at the docs for expected function definition."
        )


class MissingFunctionException(ProjectInitializationException):
    def __init__(
        self,
        func_name: str,
        script_path: Path,
    ) -> None:
        msg = f'Function "{func_name}" not defined in {script_path.resolve()}'
        super().__init__(
            msg, "Please, look at the docs for expected function definition."
        )


class ProjectRunTerminatedError(ProjectRunnerError):
    pass


class UnexpectedModelTypeException(Exception):
    def __init__(self, model_object_type: Type[Any]):
        super().__init__(
            f"train_model function returned an unsupported model object type '{model_object_type}'"
        )


class ProjectCircularDependenciesException(ProjectInitializationException):
    def __init__(self, stringified_paths: List[str]):
        super().__init__(
            self._err_msg(stringified_paths),
            "Please, check your project dependencies for errors",
        )
        self._paths: List[str] = stringified_paths

    @property
    def stringified_cycle_paths(self) -> List[str]:
        return self._paths

    @staticmethod
    def _err_msg(stringified_paths: List[str]) -> str:
        paths_as_str = "\n".join(stringified_paths)
        return f"Detected circular dependencies:\n{paths_as_str}"


class ProjectDefinitionError(ProjectInitializationException):
    pass


class ProjectFieldError(ProjectDefinitionError):
    def __init__(self, message: str, field_chain: List[str], field: str):
        self.field_chain = field_chain
        self.field = field
        self.message = message
        super().__init__(self.__str__(), "Please, fix the field value and try again.")

    def _get_field_chain(self) -> str:
        extended = self.field_chain.copy()
        extended.append(self.field)
        return ".".join(extended)

    def __str__(self) -> str:
        return f"{self.message}.\nProperty: {self._get_field_chain()}"


class NonDataFrameTypeException(Exception):
    def __init__(self, object_type: Type[Any]):
        super().__init__(
            f"function returned an unsupported object type '{object_type}'"
        )


class MissingColumnsInDataframeException(Exception):
    def __init__(self) -> None:
        super().__init__(
            "Dataframe is required to have at least two columns and must contain the primary key and the value."
        )
