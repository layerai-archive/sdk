import platform


def _is_darwin():
    return platform.system() == "Darwin"


def _is_darwin_arm64():
    # on M1 Mac platforms platform.version() returns
    # 'Darwin Kernel Version 21.5.0: Tue Apr 26 21:08:29 PDT 2022; root:xnu-8020.121.3~4/RELEASE_ARM64_T8101'
    return _is_darwin() and ("ARM64" in platform.version())


IS_DARWIN = _is_darwin()
IS_DARWIN_ARM64 = _is_darwin_arm64()
