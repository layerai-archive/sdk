import re

import pyarrow as pa
import pytest

from layer.clients.data_catalog import _get_batch_chunks


def test_get_batch_chunks_returns_single_chunk():
    data = ["aaa", "bbb", "ccc"]
    batch = pa.RecordBatch.from_arrays([pa.array(data)], names=["data"])
    chunks = [chunk.to_pydict() for chunk in _get_batch_chunks(batch)]

    assert chunks == [{"data": data}]


def test_get_batch_chunks_returns_multiple_chunks():
    data = ["a" * 3, "b" * 3, "c" * 3, "d" * 3, "e" * 2]
    batch = pa.RecordBatch.from_arrays([pa.array(data)], names=["data"])
    chunks = [
        chunk.to_pydict() for chunk in _get_batch_chunks(batch, max_chunk_size_bytes=15)
    ]

    assert chunks == [
        {"data": ["aaa", "bbb"]},
        {"data": ["ccc", "ddd"]},
        {"data": ["ee"]},
    ]


def test_get_batch_chunks_raises_when_average_row_size_exceeds_chunk_size():
    data = ["a" * 3, "b" * 3, "c" * 3, "d" * 3, "e" * 2]
    batch = pa.RecordBatch.from_arrays([pa.array(data)], names=["data"])

    expected = re.escape(
        "the average size of a single row of 6 byte(s) exceeds max chunk size of 5 byte(s)"
    )
    with pytest.raises(ValueError, match=expected):
        list(_get_batch_chunks(batch, max_chunk_size_bytes=5))


def test_get_batch_chunks_reduces_num_rows_if_chunk_size_exceeded():
    data = ["a" * 3, "b" * 3, "c" * 11, "d" * 3, "e" * 2]
    batch = pa.RecordBatch.from_arrays([pa.array(data)], names=["data"])
    chunks = [
        chunk.to_pydict() for chunk in _get_batch_chunks(batch, max_chunk_size_bytes=15)
    ]

    assert chunks == [
        {"data": ["aaa"]},
        {"data": ["bbb"]},
        {"data": ["ccccccccccc"]},
        {"data": ["ddd"]},
        {"data": ["ee"]},
    ]


def test_get_batch_chunks_raises_when_single_row_size_exceeds_chunk_size():
    data = ["a" * 3, "b" * 3, "c" * 17, "d" * 3, "e" * 2]
    batch = pa.RecordBatch.from_arrays([pa.array(data)], names=["data"])
    expected = re.escape(
        "single row in the batch at index 2 exceeds max chunk size of 15 byte(s)"
    )
    with pytest.raises(ValueError, match=expected):
        list(_get_batch_chunks(batch, max_chunk_size_bytes=15))


def test_get_batch_chunks_item_size_multiple_of_chunk_size():
    data = ["a" * 3, "b" * 3, "c" * 3, "d" * 3]
    batch = pa.RecordBatch.from_arrays([pa.array(data)], names=["data"])
    chunks = [
        chunk.to_pydict() for chunk in _get_batch_chunks(batch, max_chunk_size_bytes=14)
    ]

    assert chunks == [
        {"data": ["aaa", "bbb"]},
        {"data": ["ccc", "ddd"]},
    ]
