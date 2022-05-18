from layer.contracts.fabrics import Fabric


def test_has_member_key() -> None:
    assert Fabric.has_member_key("f-small") is True
    assert Fabric.has_member_key("invalid") is False
