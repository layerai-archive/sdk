from typing import Optional

import pytest

from layer.contracts.assets import AssetPath, AssetType


class TestCompositeAssetName:
    @pytest.mark.parametrize(
        "bad_format",
        [
            ":..124.:$",
            "org//model-name",
            "test:",
        ],
    )
    def test_parse_composite_name_with_invalid_format(self, bad_format: str) -> None:
        with pytest.raises(
            ValueError,
            match="(Please specify full path or specify asset type|Asset path does not match expected pattern)",
        ):
            AssetPath.parse(bad_format)

    @pytest.mark.parametrize(
        ("composite_name", "optional_asset_type", "expected"),
        [
            (
                "models/test",
                None,
                AssetPath(asset_name="test", asset_type=AssetType.MODEL),
            ),
            (
                "models/test",
                AssetType.MODEL,
                AssetPath(asset_name="test", asset_type=AssetType.MODEL),
            ),
            (
                "test",
                AssetType.MODEL,
                AssetPath(asset_name="test", asset_type=AssetType.MODEL),
            ),
            (
                "models/test:1",
                None,
                AssetPath(
                    asset_name="test", asset_type=AssetType.MODEL, asset_version="1"
                ),
            ),
            (
                "test:1",
                AssetType.MODEL,
                AssetPath(
                    asset_name="test", asset_type=AssetType.MODEL, asset_version="1"
                ),
            ),
            (
                "models/test:1.2",
                None,
                AssetPath(
                    asset_name="test",
                    asset_type=AssetType.MODEL,
                    asset_version="1",
                    asset_build=2,
                ),
            ),
            (
                "the-project/models/test:1.2",
                None,
                AssetPath(
                    asset_name="test",
                    asset_type=AssetType.MODEL,
                    asset_version="1",
                    asset_build=2,
                    project_name="the-project",
                ),
            ),
            (
                "the-org/the-project/models/test:1.2",
                None,
                AssetPath(
                    asset_name="test",
                    asset_type=AssetType.MODEL,
                    asset_version="1",
                    asset_build=2,
                    project_name="the-project",
                    org_name="the-org",
                ),
            ),
            (
                "The-org/The_Project/datasets/test_asset",
                None,
                AssetPath(
                    asset_name="test_asset",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                ),
            ),
            (
                "The-org/The_Project/datasets/test_asset#the_selected",
                None,
                AssetPath(
                    asset_name="test_asset",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                    asset_selector="the_selected",
                ),
            ),
            (
                "The-org/The_Project/datasets/test_asset:the_tag.12#the_selected",
                None,
                AssetPath(
                    asset_name="test_asset",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                    asset_version="the_tag",
                    asset_build=12,
                    asset_selector="the_selected",
                ),
            ),
        ],
    )
    def test_parse_composite_name_with_valid_format(
        self,
        composite_name: str,
        optional_asset_type: Optional[AssetType],
        expected: AssetPath,
    ) -> None:
        result = AssetPath.parse(
            composite_name, expected_asset_type=optional_asset_type
        )

        assert result == expected

    def test_parse_composite_name_with_missing_asset_type(self):
        with pytest.raises(
            ValueError, match="Please specify full path or specify asset type"
        ):
            AssetPath.parse("the-model:1.21")

    @pytest.mark.parametrize(
        ("composite", "expected"),
        [
            (
                AssetPath(
                    asset_name="test_asset",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                ),
                "The-org/The_Project/datasets/test_asset",
            ),
            (
                AssetPath(
                    asset_name="test_asset",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                    asset_version=12,
                ),
                "The-org/The_Project/datasets/test_asset:12",
            ),
            (
                AssetPath(
                    asset_name="test_asset",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                    asset_version=12,
                    asset_build=8,
                ),
                "The-org/The_Project/datasets/test_asset:12.8",
            ),
            (
                AssetPath(
                    asset_name="test_asset#feature",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                ),
                "The-org/The_Project/datasets/test_asset#feature",
            ),
            (
                AssetPath(
                    asset_name="test_asset#feature",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                    asset_version=12,
                ),
                "The-org/The_Project/datasets/test_asset#feature:12",
            ),
            (
                AssetPath(
                    asset_name="test_asset#feature",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                    asset_version=12,
                    asset_build=8,
                ),
                "The-org/The_Project/datasets/test_asset#feature:12.8",
            ),
        ],
    )
    def test_composite_asset_name_to_path(
        self, composite: AssetPath, expected: str
    ) -> None:
        result = composite.path()

        assert result == expected

    def test_composite_with_project_name(self) -> None:
        src = AssetPath(
            asset_name="test_asset",
            asset_type=AssetType.DATASET,
            project_name="The_Project",
            org_name="The-org",
            asset_version=12,
            asset_build=8,
        )
        result = src.with_project_name("new-project-name")

        assert result.project_name == "new-project-name"

        src_back = result.with_project_name("The_Project")
        assert src == src_back
