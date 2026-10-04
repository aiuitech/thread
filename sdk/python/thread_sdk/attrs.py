"""Attributes and filters, the Python twin of the TypeScript SDK's `attrs.ts`.

Every document carries one 32-bit `attrs` word, stored in its dense entry (slabs and delta segments, inside the scan
prefix every query reads anyway) and in its `docs` entry. A generation may declare a schema (`manifest["attrs"]`)
that packs named fields into that word; `pack_attrs` turns `{"tenant": 5, "lang": "en"}` into the word and
`compile_filter` turns `{"tenant": 5, "lang": {"$in": ["en", "fr"]}}` into the clause list the kernels apply inside
every scan, before top-k insertion (so a filtered-out entry never takes a candidate slot). Without a schema the
whole word is one numeric field named `attrs`.

Compiled form (`u32` words, `thread-core/src/filter.rs`): `[n, (mask, op, a, b, n_values, values...) x n]`, the
clauses ANDed; every clause tests `x = attrs & mask`. Both SDKs compile the same expressions to the same words
(`sdk/ts/test/fixtures/filters.json` holds the shared vectors).
"""

from __future__ import annotations

import math
import re
from array import array
from typing import Any, Mapping, Sequence

from .wire import ThreadError

__all__ = ["define_attrs", "validate_schema", "schema_extends", "pack_attrs", "unpack_attrs", "compile_filter",
           "filter_matches", "AttrSchema", "AttrFilterExpr"]

#: `{"fields": [{"name", "type": "number" | "enum" | "bool", "offset", "bits", "min"?, "values"?}, ...]}`.
AttrSchema = dict[str, Any]
#: `{field: value | {"$eq" | "$ne" | "$in" | "$nin" | "$lt" | "$lte" | "$gt" | "$gte" | "$any" | "$all" | "$none": ...}}`.
AttrFilterExpr = Mapping[str, Any]

OP_EQ, OP_NE, OP_IN, OP_NIN, OP_RANGE, OP_BITS_ANY, OP_BITS_ALL, OP_BITS_NONE = range(8)

_RAW_FIELD: dict[str, Any] = {"name": "attrs", "type": "number", "offset": 0, "bits": 32}
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_U32 = 0xFFFFFFFF


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _mask(bits: int) -> int:
    return _U32 if bits >= 32 else (1 << bits) - 1


def define_attrs(spec: Mapping[str, Any], prev: AttrSchema | None = None) -> AttrSchema:
    """Lays out named fields in declaration order into the 32-bit word. Each field is `{"bits": n, "min"?: m}` (an
    unsigned integer), `{"values": [...]}` (an enum, stored as 1..n) or `"bool"`. With `prev` (the schema in
    force), fields that already exist keep their offset and width (an enum may append values that fit its width;
    a number keeps its bits and min), fields `prev` has that `spec` omits are kept, and new fields go after them:
    stored words stay valid."""
    fields: list[dict[str, Any]] = [dict(f, **({"values": list(f["values"])} if "values" in f else {}))
                                    for f in (prev["fields"] if prev else [])]
    offset = max((f["offset"] + f["bits"] for f in fields), default=0)
    for name, s in spec.items():
        if not isinstance(name, str) or not _NAME.match(name):
            raise ThreadError(f"attrs: bad field name {name!r}", 400)
        if s == "bool" or (isinstance(s, Mapping) and s.get("bool")):
            f: dict[str, Any] = {"name": name, "type": "bool", "offset": offset, "bits": 1}
        elif isinstance(s, Mapping) and "values" in s:
            values = s["values"]
            if (not isinstance(values, (list, tuple)) or not values or any(not isinstance(v, str) for v in values)
                    or len(set(values)) != len(values)):
                raise ThreadError(f"attrs: field {name}: values must be distinct strings", 400)
            f = {"name": name, "type": "enum", "offset": offset, "bits": max(1, math.ceil(math.log2(len(values) + 1))),
                 "values": list(values)}
        elif isinstance(s, Mapping) and "bits" in s:
            bits, mn = s["bits"], s.get("min")
            if not _is_int(bits) or not 1 <= bits <= 32:
                raise ThreadError(f"attrs: field {name}: bits must be 1..32", 400)
            if mn is not None and not _is_int(mn):
                raise ThreadError(f"attrs: field {name}: min must be an integer", 400)
            f = {"name": name, "type": "number", "offset": offset, "bits": bits}
            if mn:
                f["min"] = mn
        else:
            raise ThreadError(f"attrs: field {name}: expected {{bits}}, {{values}} or \"bool\"", 400)
        at = next((i for i, p in enumerate(fields) if p["name"] == name), -1)
        if at >= 0:
            # an existing field keeps its layout
            p = fields[at]
            same = p["type"] == f["type"] and (p["type"] != "number" or (p["bits"] == f["bits"] and (p.get("min") or 0) == (f.get("min") or 0)))
            enum_ok = p["type"] != "enum" or (f["values"][: len(p["values"])] == p["values"] and len(f["values"]) <= _mask(p["bits"]))
            if not same or not enum_ok:
                raise ThreadError(f"attrs: field {name} exists with another layout (a schema only adds fields, or enum "
                                  f"values that fit the field's {p['bits']} bits)", 400)
            if "values" in f:
                fields[at] = {**p, "values": f["values"]}
            continue
        offset += f["bits"]
        fields.append(f)
    if offset > 32:
        raise ThreadError(f"attrs: the fields need {offset} bits, the word has 32", 400)
    return validate_schema({"fields": fields})


def validate_schema(schema: Any) -> AttrSchema:
    """Checks a schema read from a manifest (fields disjoint and within the word, names unique)."""
    if not isinstance(schema, Mapping) or not isinstance(schema.get("fields"), list):
        raise ThreadError("attrs: schema needs a fields array", 400)
    names: set[str] = set()
    used = 0
    for f in schema["fields"]:
        name = f.get("name") if isinstance(f, Mapping) else None
        if not isinstance(name, str) or not name or name in names:
            raise ThreadError(f"attrs: bad or duplicate field name {name!r}", 400)
        names.add(name)
        if f.get("type") not in ("number", "enum", "bool"):
            raise ThreadError(f"attrs: field {name}: unknown type {f.get('type')!r}", 400)
        off, bits = f.get("offset"), f.get("bits")
        if not _is_int(off) or not _is_int(bits) or off < 0 or bits < 1 or off + bits > 32:
            raise ThreadError(f"attrs: field {name}: bad offset/bits", 400)
        if f["type"] == "bool" and bits != 1:
            raise ThreadError(f"attrs: field {name}: a bool is one bit", 400)
        if f["type"] == "enum" and (not isinstance(f.get("values"), list) or len(f["values"]) > _mask(bits)):
            raise ThreadError(f"attrs: field {name}: enum values do not fit {bits} bits", 400)
        m = _mask(bits) << off
        if used & m:
            raise ThreadError(f"attrs: field {name} overlaps another field", 400)
        used |= m
    return dict(schema)


def schema_extends(prev: AttrSchema, nxt: AttrSchema) -> bool:
    """Whether `nxt` keeps every field of `prev` as it is (an enum may append values): the rule for changing a
    schema."""
    by_name = {f["name"]: f for f in nxt["fields"]}
    for p in prev["fields"]:
        n = by_name.get(p["name"])
        if (n is None or n["type"] != p["type"] or n["offset"] != p["offset"] or n["bits"] != p["bits"]
                or (n.get("min") or 0) != (p.get("min") or 0)):
            return False
        if p["type"] == "enum" and n.get("values", [])[: len(p.get("values", []))] != p.get("values", []):
            return False
    return True


def _field(schema: AttrSchema | None, name: str) -> dict[str, Any]:
    if schema:
        for f in schema["fields"]:
            if f["name"] == name:
                return f
    if not (schema and schema["fields"]) and name == _RAW_FIELD["name"]:
        return _RAW_FIELD
    if schema:
        raise ThreadError(f"attrs: unknown field {name!r}", 400)
    raise ThreadError('attrs: this index has no attribute schema; the only field is "attrs"', 400)


def _encode(f: dict[str, Any], v: Any) -> int | None:
    """A field value as the unsigned integer it is stored as, or None when it cannot be represented."""
    if f["type"] == "bool":
        if v is True or (v == 1 and _is_int(v)):
            return 1
        if v is False or (v == 0 and _is_int(v)):
            return 0
        return None
    if f["type"] == "enum":
        if not isinstance(v, str):
            return None
        try:
            return f["values"].index(v) + 1
        except ValueError:
            return None
    if not _is_int(v):
        return None
    x = v - (f.get("min") or 0)
    return x if 0 <= x <= _mask(f["bits"]) else None


def pack_attrs(schema: AttrSchema | None, values: Any) -> int:
    """Packs field values into the word (unset fields are 0). An int is taken as the raw word."""
    if _is_int(values):
        if not 0 <= values <= _U32:
            raise ThreadError("attrs: a raw word must be a u32", 400)
        return values
    if not isinstance(values, Mapping):
        raise ThreadError("attrs: expected a mapping of field values or a u32", 400)
    word = 0
    for name, v in values.items():
        if v is None:
            continue
        f = _field(schema, name)
        x = _encode(f, v)
        if x is None:
            extra = f" ({f['bits']} bits from {f.get('min') or 0})" if f["type"] == "number" else ""
            raise ThreadError(f"attrs: {name}: {v!r} is not a valid {f['type']}{extra}", 400)
        word |= x << f["offset"]
    return word


def unpack_attrs(schema: AttrSchema | None, word: int) -> dict[str, Any]:
    """The field values of a word under the schema (`{"attrs": word}` without one)."""
    if not (schema and schema["fields"]):
        return {"attrs": word & _U32}
    out: dict[str, Any] = {}
    for f in schema["fields"]:
        x = (word >> f["offset"]) & _mask(f["bits"])
        if f["type"] == "bool":
            out[f["name"]] = x == 1
        elif f["type"] == "enum":
            out[f["name"]] = "" if x == 0 or x > len(f["values"]) else f["values"][x - 1]
        else:
            out[f["name"]] = x + (f.get("min") or 0)
    return out


_Clause = tuple[int, int, int, int, list[int]]
_NOTHING: _Clause = (0, OP_RANGE, 1, 0, [])


def compile_filter(schema: AttrSchema | None, expr: AttrFilterExpr | None) -> array:
    """Compiles a filter expression against the schema into the kernel's words (`array('I')`, empty for no
    filter). Conditions on different fields are ANDed; within a field each operator adds a clause (`$gt`/`$gte`/
    `$lt`/`$lte` combine into one range). A value a field cannot hold makes `$eq` match nothing and `$ne` match
    everything; such values are dropped from `$in` / `$nin`."""
    if not expr:
        return array("I")
    if not isinstance(expr, Mapping):
        raise ThreadError("filter: expected a mapping of field conditions", 400)
    clauses: list[_Clause] = []
    for name, cond in expr.items():
        if not isinstance(name, str) or name.startswith("$"):
            raise ThreadError(f"filter: unsupported operator {name} at the top level (conditions are ANDed; nest "
                              "operators under a field)", 400)
        f = _field(schema, name)
        shift = f["offset"]
        fmask = _mask(f["bits"])
        mask = fmask << shift
        c = cond if isinstance(cond, Mapping) else {"$eq": cond}

        def enc(v: Any, op: str) -> int | None:
            if v is None or not isinstance(v, (int, str, bool)):
                raise ThreadError(f"filter: {name} {op}: bad value {v!r}", 400)
            x = _encode(f, v)
            return None if x is None else x << shift

        lo, hi, ranged = 0, fmask, False
        for op, v in c.items():
            if op == "$eq":
                x = enc(v, op)
                clauses.append(_NOTHING if x is None else (mask, OP_EQ, x, 0, []))
            elif op == "$ne":
                x = enc(v, op)
                if x is not None:
                    clauses.append((mask, OP_NE, x, 0, []))
            elif op in ("$in", "$nin"):
                if not isinstance(v, (list, tuple)):
                    raise ThreadError(f"filter: {name} {op}: expected a list", 400)
                xs = sorted({x for x in (enc(e, op) for e in v) if x is not None})
                if op == "$in":
                    clauses.append((mask, OP_IN, 0, 0, xs) if xs else _NOTHING)
                elif xs:
                    clauses.append((mask, OP_NIN, 0, 0, xs))
            elif op in ("$lt", "$lte", "$gt", "$gte"):
                if f["type"] != "number" or not _is_int(v):
                    raise ThreadError(f"filter: {name} {op}: needs an integer on a number field", 400)
                x = v - (f.get("min") or 0)
                ranged = True
                if op == "$lt":
                    hi = min(hi, x - 1)
                elif op == "$lte":
                    hi = min(hi, x)
                elif op == "$gt":
                    lo = max(lo, x + 1)
                else:
                    lo = max(lo, x)
            elif op in ("$any", "$all", "$none"):
                if f["type"] != "number" or not _is_int(v) or not 0 <= v <= _U32:
                    raise ThreadError(f"filter: {name} {op}: needs a u32 bit mask on a number field", 400)
                m = (v & fmask) << shift
                if m == 0:
                    if op == "$any":
                        clauses.append(_NOTHING)
                    continue
                clauses.append((m, {"$any": OP_BITS_ANY, "$all": OP_BITS_ALL, "$none": OP_BITS_NONE}[op], 0, 0, []))
            else:
                raise ThreadError(f"filter: {name}: unknown operator {op}", 400)
        if ranged:
            lo, hi = max(lo, 0), min(hi, fmask)
            if lo > hi:
                clauses.append(_NOTHING)
            elif lo > 0 or hi < fmask:
                clauses.append((mask, OP_RANGE, lo << shift, hi << shift, []))
    return clauses_to_words(clauses)


def clauses_to_words(clauses: Sequence[tuple[int, int, int, int, Sequence[int]]]) -> array:
    out = array("I")
    if not clauses:
        return out
    out.append(len(clauses))
    for mask, op, a, b, values in clauses:
        out.extend((mask & _U32, op, a & _U32, b & _U32, len(values)))
        out.extend(v & _U32 for v in values)
    return out


def filter_matches(words: Sequence[int] | None, attrs: int) -> bool:
    """Evaluates compiled words in Python (the kernel's `AttrFilter::matches` rule)."""
    if not words:
        return True
    n, pos = words[0], 1
    for _ in range(n):
        mask, op, a, b, nv = words[pos : pos + 5]
        pos += 5
        values = words[pos : pos + nv]
        pos += nv
        x = attrs & mask
        if op == OP_EQ:
            ok = x == a
        elif op == OP_NE:
            ok = x != a
        elif op == OP_IN:
            ok = x in values
        elif op == OP_NIN:
            ok = x not in values
        elif op == OP_RANGE:
            ok = a <= x <= b
        elif op == OP_BITS_ANY:
            ok = x != 0
        elif op == OP_BITS_ALL:
            ok = x == mask
        elif op == OP_BITS_NONE:
            ok = x == 0
        else:
            raise ThreadError(f"filter: unknown op {op}", 400)
        if not ok:
            return False
    return True
