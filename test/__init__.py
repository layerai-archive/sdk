def _is_darwin_arm64():
    import platform

    # on M1 Mac platforms the result should be
    # 'Darwin Kernel Version 21.5.0: Tue Apr 26 21:08:29 PDT 2022; root:xnu-8020.121.3~4/RELEASE_ARM64_T8101'
    platform_version = platform.version()
    return "Darwin" in platform_version and "ARM64" in platform_version


IS_DARWIN_ARM64 = _is_darwin_arm64()
