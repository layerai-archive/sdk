import uuid

import jwt

from layer.config import ClientConfig


class TestClientConfig:
    def test_gives_correct_organization_account_ids(self) -> None:
        personal_account_id = uuid.uuid4()
        org_account_id_1 = uuid.uuid4()
        org_account_id_2 = uuid.uuid4()
        layer_claim_prefix = "https://layer.co"
        payload = {
            f"{layer_claim_prefix}/account_id": str(personal_account_id),
            f"{layer_claim_prefix}/account_permissions": {
                str(org_account_id_1): ["read", "write"],
                str(org_account_id_2): ["read"],
            },
        }
        token = str(jwt.encode(payload, "secret", algorithm="HS256"))

        cfg = ClientConfig(access_token=token)

        ids = cfg.organization_account_ids()
        assert len(ids) == 2
        assert cfg.personal_account_id() == personal_account_id
        assert personal_account_id not in ids
