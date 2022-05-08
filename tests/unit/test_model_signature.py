import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from layer.api.value.schema_column_pb2 import SchemaColumn
from layer.mlmodels.service import MLModelService


class TestModelFlavors:
    def test_tensorflow_signature(self):
        data = pd.DataFrame(
            data={
                "clean_message": [
                    "congratulation awarded cd voucher 125gift guaranteed free entry",
                    "free entry to lottery",
                    "how are you?",
                    "ok",
                    "doing fine",
                    "you won the lottery!!",
                    "would really appreciate call need someone talk",
                    "dear good morning",
                    "ok asked money far",
                ],
                "is_spam": [1, 1, 0, 0, 0, 1, 0, 0, 0],
            }
        )
        vectorizer = CountVectorizer(lowercase=False)
        vectorizer.fit(data["clean_message"])

        X_train, X_test, y_train, y_test = train_test_split(
            data["clean_message"],
            data["is_spam"],
            test_size=0.15,
            random_state=0,
            shuffle=True,
            stratify=data["is_spam"],
        )

        X_test_trans = vectorizer.transform(X_test).toarray()

        signature = MLModelService.get_model_signature(X_test_trans, y_train)
        assert signature.input_schema.columns[0].type == SchemaColumn.DATA_TYPE_LONG
        assert signature.output_schema.columns[0].name == "is_spam"
        assert signature.output_schema.columns[0].type == SchemaColumn.DATA_TYPE_LONG
