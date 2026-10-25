# tests/test_arg_normalization.py
"""Tests for _ToolArgNormalizationMiddleware and _normalize_tool_args.

Some MCP clients never show the advertised inputSchema to the model (or the
model ignores it), so calls arrive with guessed argument names (`file_path`,
`page_numbers`) and wrong JSON types (`pages` as [1] instead of "1"). The
middleware rewrites the obviously-meant cases before pydantic validation.
"""

import asyncio
import json

import pdf_mcp.server as server
from pdf_mcp.server import _normalize_tool_args, _schema_types, _snake_case

# ── _snake_case ──────────────────────────────────────────────────────────


def test_snake_case_camel():
    assert _snake_case("filePath") == "file_path"
    assert _snake_case("pageNumbers") == "page_numbers"
    assert _snake_case("maxPages") == "max_pages"


def test_snake_case_passthrough():
    assert _snake_case("path") == "path"
    assert _snake_case("page_numbers") == "page_numbers"


def test_snake_case_acronym_run():
    assert _snake_case("filePATH") == "file_path"


# ── key renaming ─────────────────────────────────────────────────────────

_INFO_PROPS = {
    "path": {"type": "string"},
    "detail": {"type": "boolean"},
    "content_trust": {"type": "boolean"},
}

_RENDER_PROPS = {
    "path": {"type": "string"},
    "pages": {"type": "string"},
    "dpi": {"type": "integer"},
    "clip": {"anyOf": [{"type": "array"}, {"type": "null"}]},
}

_CORPUS_PROPS = {
    "paths": {"type": ["string", "array"]},
    "budget_seconds": {"type": "integer"},
}


def test_file_path_maps_to_path():
    out = _normalize_tool_args(_INFO_PROPS, {"file_path": "/a.pdf"})
    assert out == {"path": "/a.pdf"}


def test_camelcase_file_path_maps_to_path():
    out = _normalize_tool_args(_INFO_PROPS, {"filePath": "/a.pdf"})
    assert out == {"path": "/a.pdf"}


def test_page_numbers_maps_to_pages():
    out = _normalize_tool_args(_RENDER_PROPS, {"page_numbers": [1], "path": "/a.pdf"})
    assert out == {"pages": "1", "path": "/a.pdf"}


def test_singular_path_maps_to_plural_paths_on_corpus_tool():
    out = _normalize_tool_args(_CORPUS_PROPS, {"path": "/docs"})
    assert out == {"paths": "/docs"}


def test_plural_paths_maps_to_path_on_single_doc_tool():
    out = _normalize_tool_args(_INFO_PROPS, {"paths": ["/a.pdf"]})
    assert out == {"path": "/a.pdf"}


def test_page_number_maps_to_page_when_tool_takes_page():
    # pdf_extract_chart's canonical name is `page` (int), not `pages`.
    props = {"path": {"type": "string"}, "page": {"type": "integer"}}
    out = _normalize_tool_args(props, {"page_number": "3", "path": "/a.pdf"})
    assert out == {"page": 3, "path": "/a.pdf"}


def test_unknown_key_is_preserved_for_validation_error():
    out = _normalize_tool_args(_INFO_PROPS, {"bogus": 1, "path": "/a.pdf"})
    assert out == {"bogus": 1, "path": "/a.pdf"}


def test_explicit_canonical_key_is_not_overwritten_by_alias():
    # Sending both must not silently pick one: the alias key survives so
    # pydantic flags it as unexpected.
    out = _normalize_tool_args(_INFO_PROPS, {"path": "/a.pdf", "file_path": "/b.pdf"})
    assert out == {"path": "/a.pdf", "file_path": "/b.pdf"}


def test_two_aliases_to_same_param_do_not_clobber_each_other():
    out = _normalize_tool_args(
        _INFO_PROPS, {"file_path": "/a.pdf", "pdf_path": "/b.pdf"}
    )
    assert out == {"path": "/a.pdf", "pdf_path": "/b.pdf"}


def test_query_alias_maps():
    props = {"path": {"type": "string"}, "query": {"type": "string"}}
    out = _normalize_tool_args(props, {"q": "budget", "path": "/a.pdf"})
    assert out == {"query": "budget", "path": "/a.pdf"}


# ── value coercion ───────────────────────────────────────────────────────


def test_pages_list_coerces_to_range_string():
    out = _normalize_tool_args(_RENDER_PROPS, {"pages": [1, 2, 5]})
    assert out["pages"] == "1,2,5"


def test_pages_int_coerces_to_string():
    out = _normalize_tool_args(_RENDER_PROPS, {"pages": 3})
    assert out["pages"] == "3"


def test_pages_string_unchanged():
    out = _normalize_tool_args(_RENDER_PROPS, {"pages": "1-3"})
    assert out["pages"] == "1-3"


def test_path_list_takes_first_element():
    out = _normalize_tool_args(_INFO_PROPS, {"path": ["/a.pdf", "/b.pdf"]})
    assert out["path"] == "/a.pdf"


def test_integer_string_coerces_to_int():
    out = _normalize_tool_args(_RENDER_PROPS, {"dpi": "150"})
    assert out["dpi"] == 150


def test_boolean_string_coerces_to_bool():
    out = _normalize_tool_args(_INFO_PROPS, {"detail": "true"})
    assert out["detail"] is True
    out = _normalize_tool_args(_INFO_PROPS, {"content_trust": "no"})
    assert out["content_trust"] is False


def test_real_bool_not_reinterpreted():
    # isinstance(True, int) is True — bools must never hit the numeric arms.
    out = _normalize_tool_args(_RENDER_PROPS, {"dpi": True})
    assert out["dpi"] is True


def test_stringified_json_array_coerces_for_array_param():
    out = _normalize_tool_args(_RENDER_PROPS, {"clip": "[0.1, 0.2, 0.3, 0.4]"})
    assert out["clip"] == [0.1, 0.2, 0.3, 0.4]


def test_unparseable_string_left_for_validation():
    out = _normalize_tool_args(_RENDER_PROPS, {"clip": "not-json"})
    assert out["clip"] == "not-json"


def test_pages_list_with_nonsense_element_unchanged():
    out = _normalize_tool_args(_RENDER_PROPS, {"pages": [True, {"a": 1}]})
    assert out["pages"] == [True, {"a": 1}]


# ── _schema_types ────────────────────────────────────────────────────────


def test_schema_types_flat():
    assert _schema_types({"type": "string"}) == {"string"}
    assert _schema_types({"type": ["string", "array"]}) == {"string", "array"}


def test_schema_types_anyof():
    prop = {"anyOf": [{"type": "array"}, {"type": "null"}]}
    assert _schema_types(prop) == {"array", "null"}


# ── end-to-end through the MCP transport ─────────────────────────────────


def _call(tool, arguments):
    from fastmcp import Client

    async def _go():
        async with Client(server.mcp) as client:
            return await client.call_tool(tool, arguments)

    return asyncio.run(_go())


def test_pdf_info_accepts_file_path_alias_e2e(sample_pdf, isolated_server):
    result = _call("pdf_info", {"file_path": sample_pdf})
    assert result.data["page_count"] == 5


def test_pdf_read_pages_accepts_page_numbers_list_e2e(sample_pdf, isolated_server):
    result = _call("pdf_read_pages", {"path": sample_pdf, "page_numbers": [1]})
    assert "page 1 content" in json.dumps(result.data)


def test_pdf_render_pages_accepts_pages_list_e2e(sample_pdf, isolated_server):
    result = _call("pdf_render_pages", {"path": sample_pdf, "pages": [1]})
    assert not result.is_error
    summary = json.loads(result.content[0].text)
    assert summary["pages_rendered"] == [1]


def test_unmapped_argument_still_fails_e2e(sample_pdf, isolated_server):
    import pytest

    with pytest.raises(Exception):
        _call("pdf_info", {"definitely_not_a_param": sample_pdf})
