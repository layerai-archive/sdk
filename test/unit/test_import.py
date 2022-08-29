from importtime_output_wrapper import (
    get_import_time,
    import_tree_to_waterfall,
    parse_import_time,
    prune_import_depth,
)


def test_import_layer_does_not_import_heavy_libs():
    _, raw_output = get_import_time(import_cmd="layer", module_only=True)
    all_imports = prune_import_depth(parse_import_time(raw_output), depth=4)

    deps = import_tree_to_waterfall(all_imports, width=64)

    assert "torch" not in deps
    assert "xgboost" not in deps
    assert "keras" not in deps
