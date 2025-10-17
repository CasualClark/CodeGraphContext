# dart.py
from pathlib import Path
from typing import Any, Dict, Optional

from codegraphcontext.utils.debug_log import (
    debug_log, info_logger, error_logger, warning_logger, debug_logger
)

# Tree-sitter query patterns for Dart.
# These patterns try to be robust across grammar variants by using alternatives.
DART_QUERIES = {
    # ---- Functions (top-level function declarations and class/extension methods)
    "functions": r"""
        ; Top-level function declarations
        (function_declaration
            name: (identifier) @name
            parameters: (formal_parameter_part)? @params
        ) @function_node

        ; Method-like declarations inside classes/extensions/mixins
        (method_declaration
            name: (identifier) @name
            (formal_parameter_part)? @params
        ) @function_node

        ; Old/alternate node names sometimes appear in forks:
        (function_signature
            name: (identifier) @name
            (formal_parameter_part)? @params
        ) @function_node
    """,

    # ---- Classes / Mixins / Enums / Extensions
    "classes": r"""
        (class_declaration
            name: (identifier) @name
        ) @class

        (mixin_declaration
            name: (identifier) @name
        ) @class

        (enum_declaration
            name: (identifier) @name
        ) @class

        (extension_declaration
            name: (identifier)? @name
        ) @class
    """,

    # ---- Imports / Exports / Parts
    "imports": r"""
        (import_directive
            uri: (string_literal) @path
        ) @import

        (export_directive
            uri: (string_literal) @path
        ) @import

        (part_directive
            uri: (string_literal) @path
        ) @import

        (part_of_directive
            (string_literal) @path
        ) @import
    """,

    # ---- Calls (identifier foo(), member x.y(), and constructor-like Foo())
    "calls": r"""
        (call_expression
            function: (identifier) @name
        )

        (call_expression
            function: (member_expression
                property: (identifier) @name
            )
        )

        ; Constructor-like invocations (new/const or implicit)
        (object_creation_expression
            constructor: (type_identifier) @name
        )

        (constructor_invocation
            constructor: (identifier) @name
        )
    """,

    # ---- Variables (top-level & local initialized declarations)
    "variables": r"""
        (initialized_variable_declaration
            variables: (declared_identifier
                name: (identifier) @name
            )
            value: (_) @value
        )

        ; Fallback for simple variable declarations (without explicit 'value:' field)
        (variable_declaration
            name: (identifier) @name
            value: (_) @value
        )

        ; Top-level 'const' or 'final' with initialization
        (top_level_variable_declaration
            declaration: (initialized_variable_declaration
                variables: (declared_identifier
                    name: (identifier) @name
                )
                value: (_) @value
            )
        )
    """,

    # ---- Comments (for crude docstring extraction)
    "comments": r"""
        (comment) @comment
    """,
}


def is_dart_file(file_path: Path) -> bool:
    return file_path.suffix == ".dart"


class DartTreeSitterParser:
    """
    Dart/Flutter parser using tree-sitter.

    It returns a structure consistent with other CodeGraphContext language modules:
    {
      "file_path": str,
      "functions": [...],
      "classes": [...],
      "variables": [...],
      "imports": [...],
      "function_calls": [...],
      "is_dependency": bool,
      "lang": "dart",
    }
    """

    def __init__(self, generic_parser_wrapper: Any):
        self.generic_parser_wrapper = generic_parser_wrapper
        self.language_name = "dart"
        self.language = generic_parser_wrapper.language
        self.parser = generic_parser_wrapper.parser

        # Compile all queries once
        self.queries = {
            name: self.language.query(query_str)
            for name, query_str in DART_QUERIES.items()
        }

    # ---------- helpers

    def _text(self, node: Any) -> str:
        return node.text.decode("utf-8")

    def _calc_complexity(self, node: Any) -> int:
        """
        Lightweight cyclomatic complexity approximation for Dart.
        """
        complexity_nodes = {
            # control flow
            "if_statement",
            "for_statement",
            "while_statement",
            "do_statement",
            "switch_statement",
            "case_clause",
            "default_clause",
            "conditional_expression",
            # logical operators commonly appear as separate nodes
            "logical_or_expression",
            "logical_and_expression",
            "binary_expression",  # catch || and && in some grammars
            # exceptions
            "try_statement",
            "catch_clause",
        }
        count = 1

        def walk(n):
            nonlocal count
            if n.type in complexity_nodes:
                count += 1
            for c in n.children:
                walk(c)

        walk(node)
        return count

    def _leading_line_comment(self, node: Any) -> Optional[str]:
        """
        Very simple docstring via leading line/block comment just before node.
        """
        prev = node.prev_sibling
        # Allow whitespace and comments; stop on other node types
        while prev and prev.type in ("comment", "\n", " ", "metadata"):
            if prev.type == "comment":
                return self._text(prev).strip()
            prev = prev.prev_sibling
        return None

    # ---------- public API

    def parse(self, file_path: Path, is_dependency: bool = False) -> Dict[str, Any]:
        try:
            source_code = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            warning_logger(f"Failed to read {file_path}: {e}")
            return {
                "file_path": str(file_path),
                "functions": [],
                "classes": [],
                "variables": [],
                "imports": [],
                "function_calls": [],
                "is_dependency": is_dependency,
                "lang": self.language_name,
            }

        tree = self.parser.parse(bytes(source_code, "utf8"))
        root = tree.root_node

        return {
            "file_path": str(file_path),
            "functions": self._find_functions(root),
            "classes": self._find_classes(root),
            "variables": self._find_variables(root),
            "imports": self._find_imports(root),
            "function_calls": self._find_calls(root),
            "is_dependency": is_dependency,
            "lang": self.language_name,
        }

    # ---------- finders

    def _find_functions(self, root) -> list[Dict[str, Any]]:
        out = []
        q = self.queries["functions"]

        # We’ll bucket captures by the function node they belong to.
        buckets: dict[int, Dict[str, Any]] = {}

        def bucket(n: Any) -> Dict[str, Any]:
            k = id(n)
            if k not in buckets:
                buckets[k] = {"node": n, "name": None, "params": None}
            return buckets[k]

        def node_for_params(pnode: Any) -> Optional[Any]:
            # Walk upward until we hit the method/function container
            cur = pnode
            container_types = {
                "function_declaration",
                "function_signature",
                "method_declaration",
            }
            while cur:
                if cur.type in container_types:
                    return cur
                cur = cur.parent
            return None

        for node, cap in q.captures(root):
            if cap == "function_node":
                bucket(node)
            elif cap == "name":
                # climb up to the function/method node
                cur = node
                while cur and cur.type not in ("function_declaration", "function_signature", "method_declaration"):
                    cur = cur.parent
                if cur:
                    b = bucket(cur)
                    b["name"] = self._text(node)
            elif cap == "params":
                fn = node_for_params(node)
                if fn:
                    b = bucket(fn)
                    b["params"] = node

        for info in buckets.values():
            fn_node = info["node"]
            name = info["name"]
            if not name:
                # Fallback: try a direct field lookup for name
                nm = fn_node.child_by_field_name("name")
                if nm:
                    name = self._text(nm)
            if not name:
                continue

            # Extract parameter names (best-effort)
            params = []
            p = info.get("params")
            if p:
                for child in p.children:
                    if child.type == "formal_parameter_list":
                        for ch in child.children:
                            if ch.type == "normal_formal_parameter":
                                idn = ch.child_by_field_name("name")
                                if idn:
                                    params.append(self._text(idn))
                            elif ch.type == "simple_formal_parameter":
                                idn = ch.child_by_field_name("name")
                                if idn:
                                    params.append(self._text(idn))
                    elif child.type == "simple_formal_parameter":
                        idn = child.child_by_field_name("name")
                        if idn:
                            params.append(self._text(idn))

            doc = self._leading_line_comment(fn_node)

            out.append({
                "name": name,
                "line_number": fn_node.start_point[0] + 1,
                "end_line": fn_node.end_point[0] + 1,
                "args": params,
                "source": self._text(fn_node),
                "source_code": self._text(fn_node),
                "docstring": doc,
                "cyclomatic_complexity": self._calc_complexity(fn_node),
                "context": None,
                "context_type": None,
                "class_context": None,
                "decorators": [],
                "lang": self.language_name,
                "is_dependency": False,
            })
        return out

    def _find_classes(self, root) -> list[Dict[str, Any]]:
        out = []
        q = self.queries["classes"]
        for node, cap in q.captures(root):
            if cap == "class":
                name_node = node.child_by_field_name("name")
                if not name_node:
                    # Some extensions may omit a name (extension on Type)
                    name_text = "extension"
                else:
                    name_text = self._text(name_node)

                out.append({
                    "name": name_text,
                    "line_number": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "bases": [],  # Dart uses 'extends', 'implements', 'with'—can be added later if needed
                    "source": self._text(node),
                    "docstring": self._leading_line_comment(node),
                    "context": None,
                    "decorators": [],
                    "lang": self.language_name,
                    "is_dependency": False,
                })
        return out

    def _find_imports(self, root) -> list[Dict[str, Any]]:
        out = []
        q = self.queries["imports"]
        for node, cap in q.captures(root):
            if cap != "path":
                continue
            # The parent is the directive node; capture line number from it.
            parent = node.parent if node.parent is not None else node
            raw = self._text(node).strip('"\'')
            out.append({
                "name": raw,  # e.g., package:flutter/material.dart
                "full_import_name": raw,
                "line_number": parent.start_point[0] + 1,
                "alias": None,  # could parse 'as alias' later
                "lang": self.language_name,
                "is_dependency": False,
            })
        return out

    def _find_calls(self, root) -> list[Dict[str, Any]]:
        out = []
        q = self.queries["calls"]
        for node, cap in q.captures(root):
            if cap == "name":
                out.append({
                    "name": self._text(node),
                    "full_name": self._text(node.parent) if node.parent else self._text(node),
                    "line_number": node.start_point[0] + 1,
                    "args": [],
                    "inferred_obj_type": None,
                    "context": None,
                    "class_context": None,
                    "lang": self.language_name,
                    "is_dependency": False,
                })
        return out

    def _find_variables(self, root) -> list[Dict[str, Any]]:
        out = []
        q = self.queries["variables"]

        # Group by the declaration node so we can attach value/line neatly
        buckets: dict[int, Dict[str, Any]] = {}

        def bucket(n: Any) -> Dict[str, Any]:
            k = id(n)
            if k not in buckets:
                buckets[k] = {"node": n, "name": None, "value": None}
            return buckets[k]

        for node, cap in q.captures(root):
            if cap == "name":
                # climb to the nearest variable declaration
                cur = node
                while cur and cur.type not in ("initialized_variable_declaration", "variable_declaration", "top_level_variable_declaration"):
                    cur = cur.parent
                if cur:
                    b = bucket(cur)
                    b["name"] = self._text(node)
            elif cap == "value":
                cur = node
                while cur and cur.type not in ("initialized_variable_declaration", "variable_declaration", "top_level_variable_declaration"):
                    cur = cur.parent
                if cur:
                    b = bucket(cur)
                    b["value"] = self._text(node)

        for info in buckets.values():
            nm = info["name"]
            if not nm:
                continue
            decl_node = info["node"]
            out.append({
                "name": nm,
                "line_number": decl_node.start_point[0] + 1,
                "value": info.get("value"),
                "type": None,
                "context": None,
                "class_context": None,
                "lang": self.language_name,
                "is_dependency": False,
            })
        return out


def pre_scan_dart(files: list[Path], parser_wrapper) -> dict:
    """
    Pre-scan Dart files to map discovered symbols (class/enum/mixin/extension/function names)
    to their file paths. This accelerates cross-linking before full parsing.
    """
    imports_map: dict[str, list[str]] = {}

    query_str = r"""
        (class_declaration name: (identifier) @name)
        (mixin_declaration name: (identifier) @name)
        (enum_declaration  name: (identifier) @name)
        (extension_declaration name: (identifier)? @name)
        (function_declaration name: (identifier) @name)
        (method_declaration   name: (identifier) @name)
    """
    try:
        query = parser_wrapper.language.query(query_str)
    except Exception as e:
        warning_logger(f"Failed to compile Dart pre-scan query: {e}")
        return imports_map

    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            tree = parser_wrapper.parser.parse(bytes(text, "utf8"))
            for node, cap in query.captures(tree.root_node):
                if cap != "name":
                    continue
                sym = node.text.decode("utf-8")
                imports_map.setdefault(sym, [])
                fullpath = str(file_path.resolve())
                if fullpath not in imports_map[sym]:
                    imports_map[sym].append(fullpath)
        except Exception as e:
            warning_logger(f"Tree-sitter pre-scan failed for {file_path}: {e}")

    return imports_map
