from typing import Optional

import pytest

from layer.projects.asset import AssetPath, AssetType, parse_asset_path


class TestCompositeEntityName:
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
            match="(Please specify full path or specify entity type|Entity path does not match expected pattern)",
        ):
            parse_asset_path(bad_format)

    @pytest.mark.parametrize(
        ("composite_name", "optional_asset_type", "expected"),
        [
            (
                "models/test",
                None,
                AssetPath(entity_name="test", asset_type=AssetType.MODEL),
            ),
            (
                "models/test",
                AssetType.MODEL,
                AssetPath(entity_name="test", asset_type=AssetType.MODEL),
            ),
            (
                "test",
                AssetType.MODEL,
                AssetPath(entity_name="test", asset_type=AssetType.MODEL),
            ),
            (
                "models/test:1",
                None,
                AssetPath(
                    entity_name="test", asset_type=AssetType.MODEL, entity_version="1"
                ),
            ),
            (
                "test:1",
                AssetType.MODEL,
                AssetPath(
                    entity_name="test", asset_type=AssetType.MODEL, entity_version="1"
                ),
            ),
            (
                "models/test:1.2",
                None,
                AssetPath(
                    entity_name="test",
                    asset_type=AssetType.MODEL,
                    entity_version="1",
                    entity_build=2,
                ),
            ),
            (
                "the-project/models/test:1.2",
                None,
                AssetPath(
                    entity_name="test",
                    asset_type=AssetType.MODEL,
                    entity_version="1",
                    entity_build=2,
                    project_name="the-project",
                ),
            ),
            (
                "the-org/the-project/models/test:1.2",
                None,
                AssetPath(
                    entity_name="test",
                    asset_type=AssetType.MODEL,
                    entity_version="1",
                    entity_build=2,
                    project_name="the-project",
                    org_name="the-org",
                ),
            ),
            (
                "The-org/The_Project/datasets/test_entity",
                None,
                AssetPath(
                    entity_name="test_entity",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                ),
            ),
            (
                "The-org/The_Project/datasets/test_entity#the_selected",
                None,
                AssetPath(
                    entity_name="test_entity",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                    entity_selector="the_selected",
                ),
            ),
            (
                "The-org/The_Project/datasets/test_entity:the_tag.12#the_selected",
                None,
                AssetPath(
                    entity_name="test_entity",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                    entity_version="the_tag",
                    entity_build=12,
                    entity_selector="the_selected",
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
        result = parse_asset_path(
            composite_name, expected_asset_type=optional_asset_type
        )

        assert result == expected

    def test_parse_composite_name_with_missing_asset_type(self):
        with pytest.raises(
            ValueError, match="Please specify full path or specify entity type"
        ):
            parse_asset_path("the-model:1.21")

    @pytest.mark.parametrize(
        ("composite", "expected"),
        [
            (
                AssetPath(
                    entity_name="test_entity",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                ),
                "The-org/The_Project/datasets/test_entity",
            ),
            (
                AssetPath(
                    entity_name="test_entity",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                    entity_version=12,
                ),
                "The-org/The_Project/datasets/test_entity:12",
            ),
            (
                AssetPath(
                    entity_name="test_entity",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                    entity_version=12,
                    entity_build=8,
                ),
                "The-org/The_Project/datasets/test_entity:12.8",
            ),
            (
                AssetPath(
                    entity_name="test_entity#feature",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                ),
                "The-org/The_Project/datasets/test_entity#feature",
            ),
            (
                AssetPath(
                    entity_name="test_entity#feature",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                    entity_version=12,
                ),
                "The-org/The_Project/datasets/test_entity#feature:12",
            ),
            (
                AssetPath(
                    entity_name="test_entity#feature",
                    asset_type=AssetType.DATASET,
                    project_name="The_Project",
                    org_name="The-org",
                    entity_version=12,
                    entity_build=8,
                ),
                "The-org/The_Project/datasets/test_entity#feature:12.8",
            ),
        ],
    )
    def test_composite_entity_name_to_path(
        self, composite: AssetPath, expected: str
    ) -> None:
        result = composite.path()

        assert result == expected

    def test_composite_with_project_name(self) -> None:
        src = AssetPath(
            entity_name="test_entity",
            asset_type=AssetType.DATASET,
            project_name="The_Project",
            org_name="The-org",
            entity_version=12,
            entity_build=8,
        )
        result = src.with_project_name("new-project-name")

        assert result.project_name == "new-project-name"

        src_back = result.with_project_name("The_Project")
        assert src == src_back
