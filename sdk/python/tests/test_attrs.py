"""The attribute schema and filter compiler against the fixtures shared with the TypeScript SDK
(sdk/ts/test/fixtures/filters.json): both must compile the same expressions to the same words."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thread_sdk import ThreadError, _native, compile_filter, define_attrs, filter_matches, pack_attrs, unpack_attrs, validate_schema
from thread_sdk.attrs import schema_extends

FX = json.loads((Path(__file__).parent.parent.parent / "ts" / "test" / "fixtures" / "filters.json").read_text())


def test_schema_layout_and_validation():
    assert define_attrs(FX["spec"]) == FX["schema"]
    assert validate_schema(FX["schema"]) == FX["schema"]
    with pytest.raises(ThreadError, match="33 bits"):
        define_attrs({"a": {"bits": 20}, "b": {"bits": 13}})
    with pytest.raises(ThreadError, match="bits"):
        define_attrs({"a": {"bits": 0}})
    with pytest.raises(ThreadError, match="distinct"):
        define_attrs({"a": {"values": ["x", "x"]}})
    with pytest.raises(ThreadError, match="field name"):
        define_attrs({"bad name": "bool"})
    with pytest.raises(ThreadError, match="overlaps"):
        validate_schema({"fields": [{"name": "a", "type": "number", "offset": 0, "bits": 8},
                                    {"name": "b", "type": "number", "offset": 4, "bits": 8}]})
    with pytest.raises(ThreadError, match="fit"):
        validate_schema({"fields": [{"name": "a", "type": "enum", "offset": 0, "bits": 1, "values": ["x", "y"]}]})
    assert define_attrs({"e": {"values": ["only"]}})["fields"][0]["bits"] == 1
    # growth against the schema in force: existing fields keep their layout, omitted ones are kept, new ones follow
    small = define_attrs({"lang": {"values": ["en", "fr"]}, "tenant": {"bits": 8}})
    grown = define_attrs({"lang": {"values": ["en", "fr", "de"]}, "extra": "bool"}, small)
    assert [(f["name"], f["offset"], f["bits"]) for f in grown["fields"]] == [("lang", 0, 2), ("tenant", 2, 8), ("extra", 10, 1)]
    assert schema_extends(small, grown)
    with pytest.raises(ThreadError, match="fit"):
        define_attrs({"lang": {"values": ["en", "fr", "de", "it"]}}, small)
    with pytest.raises(ThreadError, match="layout"):
        define_attrs({"lang": {"values": ["fr", "en"]}}, small)
    with pytest.raises(ThreadError, match="layout"):
        define_attrs({"tenant": {"bits": 9}}, small)
    assert not schema_extends(FX["schema"], define_attrs({**FX["spec"], "lang": {"values": ["fr", "en", "de"]}}))
    assert not schema_extends(FX["schema"], define_attrs({"tenant": {"bits": 11}}))


def test_pack_unpack():
    for c in FX["pack"]:
        assert pack_attrs(FX["schema"], c["values"]) == c["word"], c
    for bad in FX["packErrors"]:
        with pytest.raises(ThreadError):
            pack_attrs(FX["schema"], bad)
    assert unpack_attrs(FX["schema"], 1040389) == {"tenant": 5, "lang": "fr", "public": True, "year": 2001}
    assert unpack_attrs(FX["schema"], 0) == {"tenant": 0, "lang": "", "public": False, "year": 1970}
    assert unpack_attrs(None, 77) == {"attrs": 77}
    assert pack_attrs(None, 77) == 77
    with pytest.raises(ThreadError, match="no attribute schema"):
        pack_attrs(None, {"tenant": 1})
    assert pack_attrs(None, {"attrs": 9}) == 9
    with pytest.raises(ThreadError):
        pack_attrs(None, True)  # a bool is not a word


def test_compile_matches_fixtures():
    for c in FX["filters"]:
        assert list(compile_filter(FX["schema"], c["expr"])) == c["words"], c["expr"]
    for c in FX["filtersRaw"]:
        assert list(compile_filter(None, c["expr"])) == c["words"], c["expr"]
    for bad in FX["filterErrors"]:
        with pytest.raises(ThreadError):
            compile_filter(FX["schema"], bad)
    assert len(compile_filter(FX["schema"], None)) == 0


def test_evaluation_matches_the_kernel():
    for c in FX["matches"]:
        words = compile_filter(FX["schema"], c["expr"])
        assert _native.filter_check(words) == words[0]
        assert [filter_matches(words, a) for a in c["words"]] == c["pass"], c["expr"]
        assert _native.filter_matches(words, c["words"]) == c["pass"], c["expr"]
    assert filter_matches(None, 123) and filter_matches([], 123)
    assert _native.filter_matches(None, [1, 2]) == [True, True]
    with pytest.raises(ValueError, match="unknown op"):
        _native.filter_check([1, 0, 9, 0, 0, 0])
