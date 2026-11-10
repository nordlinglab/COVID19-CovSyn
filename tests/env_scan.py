# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Static scans for the environment variables a shell script or a Python module reads.

test_b86_launch_env_forwarding.py uses them to check that launch_phase_d.sh forwards every
variable the Phase D chain reads into its tmux session (B86). A static scan cannot see a name
built at run time, or a mapping of the environment passed to a parameter by another name than
``env``; every other uncertainty is resolved towards reporting a read, which fails a test
loudly rather than letting a variable be dropped unseen.
"""

from __future__ import annotations

import ast
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

NAME = r"[A-Z][A-Z0-9_]*"
SHELL_SPECIAL = {"BASH_SOURCE"}
# One definition of an environment access for both Python scans: the mappings and the getters of
# os, and a mapping passed in as ``env``, as ``worker_count(os.environ)`` reads VERIFY_WORKERS.
ENVIRONMENT_MAPPINGS = {"environ", "environb"}
ENVIRONMENT_GETTERS = {"getenv", "getenvb"}
ENVIRONMENT_NAMES = ENVIRONMENT_MAPPINGS | ENVIRONMENT_GETTERS
MAPPING_PARAMETER = "env"


# --- Python ---------------------------------------------------------------------------------


def _name(node: ast.AST) -> str:
    return getattr(node, "attr", getattr(node, "id", ""))


def _environment_names(tree: ast.AST) -> dict[str, str]:
    """The names a module uses for the environment accesses of os.

    Imported aliases (``from os import environ as e``) and names bound to them or to
    ``os.environ`` by assignment, annotated assignment, walrus or parameter default, repeated
    until no new name appears. Names are not scoped, so an alias in one function also marks
    the same name elsewhere, which can only report more reads.
    """
    modules = {"os"}
    names = {name: name for name in ENVIRONMENT_NAMES}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.asname for a in node.names if a.name == "os" and a.asname}
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                if alias.name in ENVIRONMENT_NAMES:
                    names[alias.asname or alias.name] = alias.name

    def target(value: ast.AST | None) -> str | None:
        if isinstance(value, ast.Name) and value.id in names:
            return names[value.id]
        if isinstance(value, ast.Attribute) and value.attr in ENVIRONMENT_NAMES and \
                isinstance(value.value, ast.Name) and value.value.id in modules:
            return value.attr
        return None

    while True:
        bound = {}
        for node in ast.walk(tree):
            pairs = []
            if isinstance(node, ast.Assign):
                pairs = [(t, node.value) for t in node.targets]
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                pairs = [(node.target, node.value)]
            elif isinstance(node, ast.arguments):
                positional = [*node.posonlyargs, *node.args]
                pairs = list(zip(positional[len(positional) - len(node.defaults):], node.defaults,
                                 strict=True))
                keyword = zip(node.kwonlyargs, node.kw_defaults, strict=True)
                pairs += [(a, d) for a, d in keyword if d]
            for name_node, value in pairs:
                name = getattr(name_node, "id", getattr(name_node, "arg", None))
                found = target(value)
                if name and found and name not in names:
                    bound[name] = found
        if not bound:
            return names
        names.update(bound)


def environment_reads(path: Path) -> set[str]:
    """Names a module reads from the environment, found in its code, not its comments.

    Covers subscripts, ``.get``, ``.setdefault``, ``.pop``, ``in`` and ``not in`` on the
    environment or on a mapping passed in as ``env``, and ``getenv`` with the key given by
    position or as ``key=``, under any name the module imports or binds them to.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _environment_names(tree)

    def kind(node: ast.AST) -> str:
        if isinstance(node, ast.NamedExpr):
            node = node.value
        name = _name(node)
        if isinstance(node, ast.Name):
            name = aliases.get(name, name)
        if name in ENVIRONMENT_MAPPINGS or name == MAPPING_PARAMETER:
            return "mapping"
        return "getter" if name in ENVIRONMENT_GETTERS else ""

    names = set()
    for node in ast.walk(tree):
        key = None
        if isinstance(node, ast.Call):
            func = node.func
            first = node.args[0] if node.args else next(
                (k.value for k in node.keywords if k.arg == "key"), None)
            if isinstance(func, ast.Attribute) and func.attr in {"get", "setdefault", "pop"}:
                if kind(func.value) == "mapping":
                    key = first
            elif kind(func) == "getter":
                key = first
        elif isinstance(node, ast.Subscript) and kind(node.value) == "mapping":
            key = node.slice
        elif isinstance(node, ast.Compare) and \
                any(isinstance(o, (ast.In, ast.NotIn)) for o in node.ops):
            if any(kind(c) == "mapping" for c in node.comparators):
                key = node.left
        if isinstance(key, ast.Constant) and isinstance(key.value, str) and \
                re.fullmatch(NAME, key.value):
            names.add(key.value)
    return names


def mentions_environment(path: Path) -> bool:
    """Whether a module touches the environment at all.

    A copy, an alias or a mapping handed to a reader such as ``worker_count`` reads the
    environment as surely as a lookup does: no environment access of os under any name, and no
    such name as a string outside a docstring, which covers ``getattr(os, "environ")`` and
    ``vars(os)["environ"]``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _environment_names(tree)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body and isinstance(node.body[0], ast.Expr)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in aliases:
            return True
        if isinstance(node, ast.Attribute) and node.attr in ENVIRONMENT_NAMES:
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "os" and \
                any(a.name in ENVIRONMENT_NAMES or a.name == "*" for a in node.names):
            return True
        if isinstance(node, ast.Constant) and node.value in ENVIRONMENT_NAMES and \
                id(node) not in docstrings:
            return True
    return False


def import_closure(module: str, source: Path) -> set[Path]:
    """The source files under ``source`` that a module imports, directly or indirectly.

    Relative imports resolve against the importing module, and each package's ``__init__.py``
    counts, as it runs on every import of the package.
    """
    root, package_name = source.parent, source.name
    seen, todo, files = set(), [module], set()
    while todo:
        name = todo.pop()
        if name in seen or not name.startswith(package_name):
            continue
        seen.add(name)
        parts = name.split(".")
        for depth in range(1, len(parts)):
            init = root.joinpath(*parts[:depth], "__init__.py")
            if init.is_file():
                files.add(init)
        path = root / (name.replace(".", "/") + ".py")
        path = path if path.is_file() else root.joinpath(*parts, "__init__.py")
        if not path.is_file():
            continue
        files.add(path)
        base = parts if path.name == "__init__.py" else parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                prefix = base[: len(base) - node.level + 1] if node.level else []
                target = ".".join([*prefix, *(node.module or "").split(".")]).strip(".")
                todo.append(target)
                todo.extend(f"{target}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Import):
                todo.extend(a.name for a in node.names)
    return files


# --- Shell ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ShellLine:
    """One logical line of a script: a command line, or a line of a here-document body."""

    text: str
    heredoc: bool = False


def strip_comment(line: str) -> str:
    """A line without its comment, as bash reads it.

    A ``#`` starts a comment at the start of a word, outside quotes; a backslash outside single
    quotes escapes the next character.
    """
    quote = None
    index = 0
    while index < len(line):
        char = line[index]
        if quote == "'":
            quote = None if char == "'" else quote
        elif char == "\\":
            index += 1
        elif quote == '"':
            quote = None if char == '"' else quote
        elif char in "'\"":
            quote = char
        elif char == "#" and (index == 0 or line[index - 1] in " \t;&|("):
            return line[:index]
        index += 1
    return line


HEREDOC = re.compile(r"<<(-?)\s*(['\"]?)([\w.-]+)\2")


def shell_lines(text: str) -> list[ShellLine]:
    """A script's logical lines.

    Comments are dropped first and continuation lines then joined, as bash does not continue a
    comment; blank lines are left out, and here-document bodies are kept as such, since they
    hold text, not commands.
    """
    lines, pending, delimiters = [], "", []
    for raw in text.splitlines():
        if delimiters:
            strip_tabs, word = delimiters[0]
            if (raw.lstrip("\t") if strip_tabs else raw) == word:
                delimiters.pop(0)
            else:
                lines.append(ShellLine(raw, heredoc=True))
            continue
        code = strip_comment(raw)
        if code.endswith("\\") and not code.endswith("\\\\"):
            pending += code[:-1] + " "
            continue
        line = pending + code
        pending = ""
        if line.strip():
            lines.append(ShellLine(line))
        delimiters = [(m.group(1) == "-", m.group(3)) for m in HEREDOC.finditer(line)]
    if pending:
        lines.append(ShellLine(pending))
    return lines


def _fold_substitutions(line: str) -> str:
    """The line with each arithmetic expansion, command substitution and backtick group as X."""
    patterns = (r"\$\(\([^()]*\)\)", r"\$\([^()]*\)", r"`[^`]*`")
    while True:
        folded = line
        for pattern in patterns:
            folded = re.sub(pattern, "X", folded)
        if folded == line:
            return line
        line = folded


REDIRECTION = re.compile(r"^\d*(?:[<>]|&>)")


def commands(line: str) -> list[list[str]]:
    """The commands of a line as lists of words, without redirections.

    Commands split at ``;``, ``&&``, ``||``, ``|`` and ``&``; an unparsable line gives none.
    """
    lexer = shlex.shlex(_fold_substitutions(line), posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        words = list(lexer)
    except ValueError:
        return []
    result, current, skip_target = [], [], False
    for word in words:
        if word and set(word) <= set(";&|"):
            result.append(current)
            current = []
        elif skip_target:
            skip_target = False
        elif REDIRECTION.match(word):
            skip_target = re.fullmatch(r"\d*(?:[<>]{1,2}|&>)", word) is not None
        else:
            current.append(word)
    return [c for c in [*result, current] if c]


OPENERS = {"if", "for", "while", "until", "case", "select"}
CLOSERS = {"fi", "done", "esac"}


def block_change(line: str) -> int:
    """How far a line moves the block depth.

    +1 for each if, loop, case or brace group (a function body included) it opens, -1 for each
    it closes.
    """
    change = 0
    for command in commands(line):
        change += (command[0] in OPENERS) - (command[0] in CLOSERS)
        change += command.count("{") - command.count("}")
    return change


def assignment(line: str) -> str | None:
    """The name a line assigns, when NAME=value is the whole of its first command.

    NAME=value followed by a word sets NAME for that command only.
    """
    parsed = commands(line)
    if parsed and len(parsed[0]) == 1:
        match = re.fullmatch(rf"({NAME})=.*", parsed[0][0], re.DOTALL)
        if match:
            return match.group(1)
    return None


def shell_variables(lines: list[ShellLine]) -> tuple[set[str], set[str], set[str]]:
    """The names a script references, assigns and reads with a ``${NAME:-}`` default.

    A name counts as assigned only when the script assigns it outside every block, so the
    assignment always runs, before it first uses it; a script that uses a name first, or
    assigns it only in a block, reads it from the environment.
    """
    referenced, defaulted, first_use, first_assignment = set(), set(), {}, {}
    depth = 0
    for number, line in enumerate(lines):
        for name in re.findall(rf"\$\{{?({NAME})", line.text):
            referenced.add(name)
            first_use.setdefault(name, number)
        defaulted |= set(re.findall(rf"\$\{{({NAME}):-", line.text))
        if line.heredoc:
            continue
        name = assignment(line.text) if depth == 0 else None
        if name:
            first_assignment.setdefault(name, number)
        depth = max(depth + block_change(line.text), 0)
    assigned = {name for name, number in first_assignment.items()
                if number < first_use.get(name, len(lines))}
    return referenced - SHELL_SPECIAL, assigned, defaulted


def environment_variables(path: Path) -> set[str]:
    """The names a script reads from the environment: defaulted, or used before assigned."""
    referenced, assigned, defaulted = shell_variables(shell_lines(path.read_text()))
    return (referenced - assigned) | defaulted


SCRIPT_CALL = re.compile(
    r"(?:^|[\s;&|(])(?:(?:bash|sh|source|\.)\s+)?[\"']?(?:\$\{?ROOT\}?/|\./)?scripts/([\w.-]+\.sh)")


def called_scripts(lines: list[ShellLine]) -> set[str]:
    """The scripts under scripts/ a script runs, however it calls them."""
    return {name for line in lines if not line.heredoc for name in SCRIPT_CALL.findall(line.text)}
