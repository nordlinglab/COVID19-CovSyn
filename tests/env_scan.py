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
# Methods that copy the environment or view it, which read it as surely as a lookup.
COPIES = {"copy", "keys", "items", "values"}
COPY_FUNCTIONS = {"copy", "deepcopy"}


# --- Python ---------------------------------------------------------------------------------


def _name(node: ast.AST) -> str:
    return getattr(node, "attr", getattr(node, "id", ""))


def copied(node: ast.AST) -> list[ast.AST]:
    """The expressions a copy or view is made of; none for anything else.

    Covers ``dict(e)``, ``dict(**e)``, ``{**e}``, ``copy.copy(e)``, ``e.copy()`` and
    ``e.keys()``.
    """
    if isinstance(node, ast.Dict):
        return [v for k, v in zip(node.keys, node.values, strict=True) if k is None]
    if not isinstance(node, ast.Call):
        return []
    func = node.func
    if _name(func) == "dict" or (isinstance(func, ast.Attribute) and
                                 _name(func.value) == "copy" and func.attr in COPY_FUNCTIONS):
        return [*node.args, *(k.value for k in node.keywords if k.arg is None)]
    if isinstance(func, ast.Attribute) and func.attr in COPIES:
        return [func.value]
    return []


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
        # A copy or view of the environment reads it as surely as the environment itself.
        if isinstance(value, (ast.Call, ast.Dict)):
            return next((t for t in map(target, copied(value)) if t), None)
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
                for t in node.targets:
                    if isinstance(t, (ast.Tuple, ast.List)) and \
                            isinstance(node.value, (ast.Tuple, ast.List)) and \
                            len(t.elts) == len(node.value.elts):
                        pairs += list(zip(t.elts, node.value.elts, strict=True))
                    else:
                        pairs.append((t, node.value))
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
    environment, a copy or view of it, or a mapping passed in as ``env``, and ``getenv`` with
    the key given by position or as ``key=``, under any name the module imports or binds them
    to.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _environment_names(tree)

    def kind(node: ast.AST) -> str:
        if isinstance(node, ast.NamedExpr):
            node = node.value
        if isinstance(node, (ast.Call, ast.Dict)):
            return "mapping" if any(kind(i) == "mapping" for i in copied(node)) else ""
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


def _scan_quotes(text: str) -> tuple[str, str | None]:
    """A text without its comments, and the quote still open at its end, as bash reads it.

    A ``#`` starts a comment at the start of a word, outside quotes, and runs to the end of its
    line; a backslash outside single quotes escapes the next character, also inside ``$'...'``.
    """
    quote, out, index = None, [], 0
    while index < len(text):
        char = text[index]
        if quote == "'":
            quote = None if char == "'" else quote
        elif quote is None and text.startswith("$'", index):
            quote = "$'"                        # ANSI-C quoting, where \' escapes a quote
            out.append("$'")
            index += 2
            continue
        elif quote == "$'" and char == "'":
            quote = None
        elif char == "\\":
            out.append(text[index:index + 2])
            index += 2
            continue
        elif quote == '"':
            quote = None if char == '"' else quote
        elif char in "'\"":
            quote = char
        elif char == "#" and (index == 0 or text[index - 1] in " \t\n;&|("):
            end = text.find("\n", index)
            index = len(text) if end < 0 else end
            continue
        out.append(char)
        index += 1
    return "".join(out), quote


def strip_comment(line: str) -> str:
    """A line without its comment, as bash reads it."""
    return _scan_quotes(line)[0]


def _trailing_backslashes(text: str) -> int:
    return len(text) - len(text.rstrip("\\"))


def _raw_words(line: str) -> list[str]:
    """A line's words with their quotes kept, substitutions folded; unparsable gives none."""
    lexer = shlex.shlex(_fold_substitutions(line), posix=False, punctuation_chars=";&|")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        return []


def _heredoc_delimiters(line: str) -> list[tuple[bool, str]]:
    """The here-documents a line opens: whether tabs are stripped, and the delimiter.

    ``<<<`` is a here-string and ``<<`` inside quotes or arithmetic is no here-document.
    """
    words, found = _raw_words(line), []
    for index, word in enumerate(words):
        match = re.search(r"(?<!<)<<(?!<)(-?)(.*)$", word)
        if not match or word[0] in "'\"":
            continue
        delimiter = match.group(2) or (words[index + 1] if index + 1 < len(words) else "")
        delimiter = delimiter.replace("\\", "").strip("'\"")
        if delimiter:
            found.append((match.group(1) == "-", delimiter))
    return found


def shell_lines(text: str) -> list[ShellLine]:
    """A script's logical lines.

    A line still inside a quote at its end goes on with the next, comments are dropped before
    continuation lines are joined (bash does not continue a comment), an odd run of trailing
    backslashes continues a line, blank lines are left out, and here-document bodies are kept
    as such, since they hold text, not commands.
    """
    lines, buffer, delimiters = [], "", []
    for raw in text.splitlines():
        if delimiters:
            strip_tabs, word = delimiters[0]
            if (raw.lstrip("\t") if strip_tabs else raw) == word:
                delimiters.pop(0)
            else:
                lines.append(ShellLine(raw, heredoc=True))
            continue
        code, open_quote = _scan_quotes(buffer + raw)
        if open_quote:
            buffer += raw + "\n"
            continue
        if _trailing_backslashes(code) % 2:
            buffer = code[:-1] + " "
            continue
        buffer = ""
        if code.strip():
            lines.append(ShellLine(code))
            delimiters = _heredoc_delimiters(code)
    if buffer.strip():
        lines.append(ShellLine(buffer))
    return lines


def _fold_substitutions(line: str) -> str:
    """The line with each substitution as X.

    Arithmetic expansions and commands, command substitutions and backtick groups.
    """
    patterns = (r"\$?\(\([^()]*\)\)", r"\$\([^()]*\)", r"`[^`]*`")
    while True:
        folded = line
        for pattern in patterns:
            folded = re.sub(pattern, "X", folded)
        if folded == line:
            return line
        line = folded


REDIRECTION = re.compile(r"^\d*(?:[<>]|&>)")


def commands(line: str) -> list[list[str]]:
    """The commands of a line as lists of words, quotes removed and redirections dropped.

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
# Reserved words after which a command, and so an opener, may follow on the same command.
LEADERS = {"then", "else", "elif", "do", "!", "time"}


def block_change(line: str) -> int:
    """How far a line moves the block depth.

    +1 for each if, loop, case, brace group (a function body included) or subshell it opens,
    -1 for each it closes; a parenthesis left open at the end of a word, as in ``ARR=(`` or
    ``$(``, opens one too. An opener counts wherever a command may start, as in ``else if``
    or after a case pattern; quoted braces and parentheses are text. A subshell closed on a
    later word of its own line, as in ``(cd x && y)``, leaves the depth raised, which can only
    report more reads.
    """
    change, starts_command = 0, True
    for word in _raw_words(line):
        if word and set(word) <= set(";&|"):
            starts_command = True
            continue
        function = word == "()" or word.endswith("()")
        if word == "{" or (not function and (word == "(" or word.endswith("(") or
                                            (word.startswith("(") and not word.startswith("((")))):
            change += 1                     # a group, subshell, array or open substitution
            starts_command = True
            continue
        if word in {"}", ")"}:
            change -= 1
            continue
        if starts_command:
            change += (word in OPENERS) - (word in CLOSERS)
            starts_command = word in LEADERS
        if word.endswith(")") and not word.startswith("(") and not function:
            starts_command = True           # a case pattern; a command follows it
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


def references(text: str) -> set[str]:
    """The names a line reads.

    ``$NAME``, ``${NAME...}``, ``${#NAME}``, ``${!NAME}`` and the bare names of arithmetic,
    ``$(( ))`` and ``(( ))``.
    """
    names = set(re.findall(rf"\$\{{?[#!]?({NAME})", text))
    for expression in re.findall(r"\(\((.*?)\)\)", text):
        names |= set(re.findall(rf"\b({NAME})\b", expression))
    return names


def shell_variables(lines: list[ShellLine]) -> tuple[set[str], set[str], set[str]]:
    """The names a script references, assigns and reads with a ``${NAME:-}`` default.

    A name counts as assigned only when the script assigns it outside every block, so the
    assignment always runs, before it first uses it; a script that uses a name first, or
    assigns it only in a block, reads it from the environment.
    """
    referenced, defaulted, first_use, first_assignment = set(), set(), {}, {}
    depth = 0
    for number, line in enumerate(lines):
        for name in references(line.text):
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


SCRIPT_PATH = re.compile(r"(?:^|/)scripts/([\w.-]+\.sh)$")
SHELLS = {"bash", "sh", "source", "."}


def script_calls(line: str) -> list[tuple[str, list[str]]]:
    """The scripts under scripts/ a line runs, each with its command's NAME=value prefix.

    Quotes are removed from the prefix words.
    A script runs when its path is a command's first word after its assignments, or the first
    word after bash, sh, source or ``.``; a path elsewhere, as in an echo, is text.
    """
    calls = []
    for command in commands(line):
        index = 0
        while index < len(command) and re.match(rf"{NAME}=", command[index]):
            index += 1
        prefix, rest = command[:index], command[index:]
        if rest and rest[0] in SHELLS:
            rest = [w for w in rest[1:] if not w.startswith("-")]
        match = SCRIPT_PATH.search(rest[0]) if rest else None
        if match:
            calls.append((match.group(1), prefix))
    return calls


def called_scripts(lines: list[ShellLine]) -> set[str]:
    """The scripts under scripts/ a script runs, however it calls them.

    Here-document bodies count too, as one may be piped to a shell.
    """
    return {name for line in lines for name, _ in script_calls(line.text)}
