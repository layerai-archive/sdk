from layer.fabric import Fabric


def test_has_member_key():
    assert Fabric.has_member_key("f-small") is True
    assert Fabric.has_member_key("invalid") is False
