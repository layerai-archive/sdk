from typing import Optional

import pytest
from yarl import URL

from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.project_full_name import ProjectFullName


class TestAssetPath:
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

    def test_parse_composite_name_with_missing_asset_type(self) -> None:
        with pytest.raises(
            ValueError, match="Please specify full path or specify asset type"
        ):
            AssetPath.parse("the-model:1.21")

    @pytest.mark.parametrize(
        ("composite", "expected_path", "expected_url_path"),
        [
            (
                AssetPath(
                    asset_name="test_asset",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                ),
                "The-org/The_Project/datasets/test_asset",
                "The-org/The_Project/datasets/test_asset",
            ),
            (
                AssetPath(
                    asset_name="test_asset",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                    asset_version="12",
                ),
                "The-org/The_Project/datasets/test_asset:12",
                "The-org/The_Project/datasets/test_asset?v=12",
            ),
            (
                AssetPath(
                    asset_name="test_asset",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                    asset_version="12",
                    asset_build=8,
                ),
                "The-org/The_Project/datasets/test_asset:12.8",
                "The-org/The_Project/datasets/test_asset?v=12.8",
            ),
            (
                AssetPath(
                    asset_name="test_asset",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                    asset_selector="feature",
                ),
                "The-org/The_Project/datasets/test_asset#feature",
                "The-org/The_Project/datasets/test_asset#feature",
            ),
            (
                AssetPath(
                    asset_name="test_asset",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                    asset_version="12",
                    asset_selector="feature",
                ),
                "The-org/The_Project/datasets/test_asset:12#feature",
                "The-org/The_Project/datasets/test_asset?v=12#feature",
            ),
            (
                AssetPath(
                    asset_name="test_asset",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                    asset_version="12",
                    asset_build=8,
                    asset_selector="feature",
                ),
                "The-org/The_Project/datasets/test_asset:12.8#feature",
                "The-org/The_Project/datasets/test_asset?v=12.8#feature",
            ),
        ],
    )
    def test_composite_asset_name_to_path_and_url(
        self, composite: AssetPath, expected_path: str, expected_url_path: str
    ) -> None:
        base_url = URL("https://layer.ai")

        assert composite.path() == expected_path
        assert str(composite.url(base_url)) == str(base_url) + "/" + expected_url_path

    def test_composite_with_project_full_name(self) -> None:
        src = AssetPath(
            asset_name="test_asset",
            asset_type=AssetType.DATASET,
            project_name="The_Project",
            org_name="The-org",
            asset_version="12",
            asset_build=8,
        )
        result = src.with_project_full_name(
            ProjectFullName(
                project_name="new-project-name",
                account_name="new-account-name",
            )
        )

        assert result.org_name == "new-account-name"
        assert result.project_name == "new-project-name"

        src_back = result.with_project_full_name(
            ProjectFullName(
                project_name="The_Project",
                account_name="The-org",
            )
        )
        assert src == src_back
