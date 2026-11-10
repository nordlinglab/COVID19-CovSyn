# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Static scans for the environment variables a shell script or a Python module reads.

test_b86_launch_env_forwarding.py uses them to check that launch_phase_d.sh forwards every
variable the Phase D chain reads into its tmux session (B86). A static scan cannot see a name
built at run time, or a mapping of the environment passed to a parameter by another name than
``env``; every other uncertainty is resolved towards reporting a read or a script call, which
fails a test loudly rather than letting a variable be dropped unseen.
"""

from __future__ import annotations

import ast
import posixpath
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

NAME = r"[A-Z][A-Z0-9_]*"
SHELL_SPECIAL = {"BASH_SOURCE"}

# --- Python ---------------------------------------------------------------------------------

# One definition of an environment access for both Python scans: the mappings and the getters of
# os, copies and views of the mappings, and a mapping passed in as ``env``, as
# ``worker_count(os.environ)`` reads VERIFY_WORKERS.
ENVIRONMENT_MAPPINGS = {"environ", "environb"}
ENVIRONMENT_GETTERS = {"getenv", "getenvb"}
ENVIRONMENT_NAMES = ENVIRONMENT_MAPPINGS | ENVIRONMENT_GETTERS
MAPPING_PARAMETER = "env"
VIEW_METHODS = {"copy", "keys", "items", "values"}
COPY_FUNCTIONS = {"copy", "deepcopy"}


def _name(node: ast.AST) -> str:
    return getattr(node, "attr", getattr(node, "id", ""))


class _Environment:
    """What in one module stands for the environment of os.

    Built once per module: the names os and copy are imported as, the names bound to the
    environment (imported, assigned, annotated, walrus, parameter default, tuple or list
    unpacking, repeated until no new name appears), and the copies and views of it. Names are
    not scoped, so an alias in one function also marks the same name elsewhere, which can only
    report more reads.
    """

    def __init__(self, tree: ast.AST) -> None:
        self.os_modules, self.copy_modules, self.copy_functions = {"os"}, {"copy"}, set()
        self.names = {name: name for name in ENVIRONMENT_NAMES}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "os":
                        self.os_modules.add(alias.asname or "os")
                    elif alias.name == "copy":
                        self.copy_modules.add(alias.asname or "copy")
            elif isinstance(node, ast.ImportFrom) and node.module in {"os", "copy"}:
                for alias in node.names:
                    if node.module == "os" and alias.name in ENVIRONMENT_NAMES:
                        self.names[alias.asname or alias.name] = alias.name
                    elif node.module == "copy" and alias.name in COPY_FUNCTIONS:
                        self.copy_functions.add(alias.asname or alias.name)
        while bound := self._new_bindings(tree):
            self.names.update(bound)

    def copied(self, node: ast.AST) -> list[ast.AST]:
        """The expressions a copy or view is made of; none for anything else.

        Covers ``dict(e)``, ``dict(**e)``, ``{**e}``, ``copy.copy(e)`` and ``deepcopy(e)``
        under any import name, ``e.copy()`` and ``e.keys()``.
        """
        if isinstance(node, ast.Dict):
            return [v for k, v in zip(node.keys, node.values, strict=True) if k is None]
        if not isinstance(node, ast.Call):
            return []
        func = node.func
        copies = _name(func) == "dict" or (isinstance(func, ast.Name) and
                                           func.id in self.copy_functions)
        copies |= isinstance(func, ast.Attribute) and func.attr in COPY_FUNCTIONS and \
            isinstance(func.value, ast.Name) and func.value.id in self.copy_modules
        if copies:
            return [*node.args, *(k.value for k in node.keywords if k.arg is None)]
        if isinstance(func, ast.Attribute) and func.attr in VIEW_METHODS:
            return [func.value]
        return []

    def stands_for(self, node: ast.AST | None, read: bool = False) -> str | None:
        """Which access of os an expression stands for, or None.

        A name binds only to the environment of os itself, so an unrelated ``config.environ``
        makes no alias. A read is judged more widely, so that none is missed: any attribute
        named like an access of os (``sys.modules["os"].environ``), and a name or attribute
        ``env``, a mapping of the environment passed along, stand for the environment.
        """
        if isinstance(node, ast.NamedExpr):
            node = node.value
        if isinstance(node, (ast.Call, ast.Dict)):
            return next((s for s in (self.stands_for(i, read)
                                     for i in self.copied(node)) if s), None)
        if read and _name(node) == MAPPING_PARAMETER:
            return "environ"
        if isinstance(node, ast.Name):
            return self.names.get(node.id)
        if isinstance(node, ast.Attribute) and node.attr in ENVIRONMENT_NAMES and (
                read or isinstance(node.value, ast.Name) and node.value.id in self.os_modules):
            return node.attr
        return None

    def _is_parameter(self, node: ast.AST) -> bool:
        """Whether an expression is the ``env`` mapping passed along, or a copy of it."""
        return _name(node) == MAPPING_PARAMETER or \
            any(self._is_parameter(i) for i in self.copied(node))

    def _new_bindings(self, tree: ast.AST) -> dict[str, str]:
        bound = {}
        for node in ast.walk(tree):
            pairs = []
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, (ast.Tuple, ast.List)) and \
                            isinstance(node.value, (ast.Tuple, ast.List)) and \
                            len(target.elts) == len(node.value.elts):
                        pairs += list(zip(target.elts, node.value.elts, strict=True))
                    else:
                        pairs.append((target, node.value))
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                pairs = [(node.target, node.value)]
            elif isinstance(node, ast.arguments):
                positional = [*node.posonlyargs, *node.args]
                pairs = list(zip(positional[len(positional) - len(node.defaults):],
                                 node.defaults, strict=True))
                keyword = zip(node.kwonlyargs, node.kw_defaults, strict=True)
                pairs += [(a, d) for a, d in keyword if d]
            for name_node, value in pairs:
                name = getattr(name_node, "id", getattr(name_node, "arg", None))
                found = self.stands_for(value) or (
                    "environ" if self._is_parameter(value) else None)
                if name and found and name not in self.names:
                    bound[name] = found
        return bound


def environment_reads(path: Path) -> set[str]:
    """Names a module reads from the environment, found in its code, not its comments.

    Covers subscripts, ``.get``, ``.setdefault``, ``.pop``, ``in`` and ``not in`` on the
    environment, a copy or view of it, or a mapping passed in as ``env``, and ``getenv`` with
    the key given by position or as ``key=``, under any name the module imports or binds them
    to.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    environment = _Environment(tree)

    def is_mapping(node: ast.AST) -> bool:
        return environment.stands_for(node, read=True) in ENVIRONMENT_MAPPINGS

    names = set()
    for node in ast.walk(tree):
        key = None
        if isinstance(node, ast.Call):
            func = node.func
            first = node.args[0] if node.args else next(
                (k.value for k in node.keywords if k.arg == "key"), None)
            if isinstance(func, ast.Attribute) and func.attr in {"get", "setdefault", "pop"}:
                key = first if is_mapping(func.value) else None
            elif environment.stands_for(func, read=True) in ENVIRONMENT_GETTERS:
                key = first
        elif isinstance(node, ast.Subscript) and is_mapping(node.value):
            key = node.slice
        elif isinstance(node, ast.Compare) and \
                any(isinstance(o, (ast.In, ast.NotIn)) for o in node.ops):
            if any(is_mapping(c) for c in node.comparators):
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
    environment = _Environment(tree)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body and isinstance(node.body[0], ast.Expr)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in environment.names:
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


def _contexts(text: str) -> list[tuple[str, int]]:
    """For each character of a text, the bash context it stands in, and the context depth.

    One walker for every quoting rule of the scans. The contexts are ``top`` (plain code),
    ``'``, ``"``, ``$'``, ```` ` ```` and the inside of ``$(`` and ``${``; a ``$(`` opens plain
    code again wherever it stands, inside double quotes or ``${...}`` too. A backslash escapes
    the next character outside single quotes, and so does it inside ``$'...'``. A character
    that opens or closes a context belongs to the outer one.
    """
    stack, parens, result, index = ["top"], [0], [], 0

    def push(kind: str, width: int) -> None:
        nonlocal index
        result.extend([(stack[-1], len(stack))] * width)
        stack.append(kind)
        parens.append(0)
        index += width

    while index < len(text):
        kind, char = stack[-1], text[index]
        plain = kind in {"top", "$(", "${"}
        if kind == "'" and char != "'" or kind == "$'" and char not in "\\'":
            result.append((kind, len(stack)))
        elif char == "\\" and kind != "'":
            result.extend([(kind, len(stack))] * len(text[index:index + 2]))
            index += 2
            continue
        elif (kind in {"'", "$'"} and char == "'") or (kind == '"' and char == '"') or \
                (kind == "`" and char == "`") or (kind == "${" and char == "}") or \
                (kind == "$(" and char == ")" and parens[-1] == 0):
            stack.pop()
            parens.pop()
            result.append((stack[-1], len(stack)))
        elif kind in {"top", "$(", "${", '"', "`"} and text.startswith("$(", index):
            push("$(", 2)
            continue
        elif kind in {"top", "$(", "${", '"'} and text.startswith("${", index):
            push("${", 2)
            continue
        elif plain and text.startswith("$'", index):
            push("$'", 2)
            continue
        elif (plain and char in "'\"`") or (kind == '"' and char == "`"):
            push(char, 1)
            continue
        else:
            if kind == "$(" and char in "()":
                parens[-1] += 1 if char == "(" else -1
            result.append((kind, len(stack)))
        index += 1
    return result


def _open_context(text: str) -> str | None:
    """The quote or substitution still open at the end of a text, or None."""
    contexts = _contexts(text + " ")
    kind = contexts[-1][0] if contexts else "top"
    return None if kind == "top" else kind


def _scan_quotes(text: str) -> tuple[str, str | None]:
    """A text without its comments, and the quote still open at its end, as bash reads it.

    A ``#`` starts a comment at the start of a word in plain code and runs to the end of its
    line; quoting follows _contexts.
    """
    contexts, out, index = _contexts(text), [], 0
    while index < len(text):
        char = text[index]
        if char == "#" and contexts[index][0] in {"top", "$("} and \
                (index == 0 or text[index - 1] in " \t\n;&|("):
            end = text.find("\n", index)
            index = len(text) if end < 0 else end
            continue
        out.append(char)
        index += 1
    open_kind = _open_context("".join(out))
    return "".join(out), open_kind if open_kind in {"'", '"', "$'", "`"} else None


def strip_comment(line: str) -> str:
    """A line without its comment, as bash reads it."""
    return _scan_quotes(line)[0]


def _trailing_backslashes(text: str) -> int:
    return len(text) - len(text.rstrip("\\"))


def _closing_parenthesis(line: str, start: int) -> int:
    """The index of the ``)`` closing the ``(`` at ``start``, or -1 if it stays open.

    The ``(`` is that of a ``$(``, ``$((`` or ``((`` at ``start - 1`` or ``start``; quoting
    follows _contexts. A group holding a case statement, whose patterns end in an unpaired
    ``)``, is taken as open, which leaves it unfolded.
    """
    opener = start - 1 if start and line[start - 1] == "$" else start
    contexts = _contexts(line)
    depth = contexts[opener][1]
    if line.startswith("$(", opener):
        # The substitution's own context closes on its matching parenthesis.
        for index in range(start + 1, len(line)):
            if contexts[index][1] == depth and line[index] == ")":
                return -1 if re.search(r"\bcase\b", line[start:index]) else index
        return -1
    level = 0
    for index in range(start, len(line)):
        if contexts[index][1] == depth and contexts[index][0] in {"top", "$(", "${"}:
            level += {"(": 1, ")": -1}.get(line[index], 0)
            if level == 0:
                return -1 if re.search(r"\bcase\b", line[start:index]) else index
    return -1


def _fold_balanced(line: str, opener: str) -> str:
    """The line with each closed ``opener ... )`` group as X; a group left open is kept."""
    start = line.find(opener)
    while start >= 0:
        end = _closing_parenthesis(line, start + opener.index("("))
        if end < 0:
            start = line.find(opener, start + 1)
            continue
        line = line[:start] + "X" + line[end + 1:]
        start = line.find(opener, start)
    return line


def _fold_arithmetic(line: str) -> str:
    """The line with each arithmetic expansion and command as X, ``$[ ]`` included."""
    line = re.sub(r"\$\[[^\]]*\]", "X", line)
    for opener in ("$((", "(("):
        line = _fold_balanced(line, opener)
    return line


def _fold_substitutions(line: str) -> str:
    """The line with each substitution as X.

    Arithmetic expansions and commands, command substitutions and backtick groups, nested
    parentheses included; one left open stays as it is.
    """
    return _fold_balanced(_fold_arithmetic(re.sub(r"`[^`]*`", "X", line)), "$(")


def _raw_words(line: str) -> list[str]:
    """A line's words with their quotes kept, substitutions folded; unparsable gives none."""
    lexer = shlex.shlex(_fold_substitutions(line), posix=False, punctuation_chars=";&|")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        return []


def _plain_positions(line: str) -> list[bool]:
    """For each character of a line, whether it stands in plain code.

    Plain code is outside quotes and ``${...}``, or inside a ``$(...)`` wherever that stands;
    only there does bash take ``<<`` for a redirection.
    """
    return [kind in {"top", "$("} for kind, _ in _contexts(line)]


def _shell_word(text: str) -> str | None:
    """The first shell word of a text, quotes and backslashes removed; None if there is none.

    That is how bash reads a here-document delimiter; an unparsable word gives None too.
    """
    text = text.lstrip(" \t")
    contexts, end = _contexts(text), 0
    while end < len(text) and not (contexts[end] == ("top", 1) and
                                   (text[end].isspace() or text[end] in ";&|<>()")):
        end += 1
    try:
        words = shlex.split(text[:end])
    except ValueError:
        return None
    return words[0] if words else None


def _heredoc_delimiters(line: str) -> list[tuple[bool, str]]:
    """The here-documents a line opens: whether tabs are stripped, and the delimiter.

    ``<<`` opens one wherever bash takes it for a redirection: in plain code, inside a command
    substitution too, even within double quotes or ``${...}``, and in ``let x=1<<2`` as bash
    does (checked with bash 5); not inside quotes, ``${...}`` or arithmetic. ``<<<`` is a
    here-string. The delimiter is the next shell word with its quotes and backslashes removed.
    """
    if "<<" not in line:
        return []
    folded = _fold_arithmetic(line)
    plain, found = _plain_positions(folded), []
    for match in re.finditer(r"(?<!<)<<(?!<)(-?)", folded):
        if plain[match.start()]:
            delimiter = _shell_word(folded[match.end():])
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

    +1 for each if, loop, case, brace group (a function body included, ``f(){`` too) or
    subshell it opens, -1 for each it closes; a parenthesis left open at the end of a word, as
    in ``ARR=(`` or ``$(``, opens one too. An opener counts wherever a command may start, as
    in ``else if`` or after a case pattern; quoted braces and parentheses are text. A subshell
    closed on a later word of its own line, as in ``(cd x && y)``, leaves the depth raised,
    which can only report more reads.
    """
    change, starts_command = 0, True
    for word in _raw_words(line):
        if word and set(word) <= set(";&|"):
            starts_command = True
            continue
        function = word == "()" or word.endswith("()")
        if word == "{" or word.endswith("){") or (
                not function and (word == "(" or word.endswith("(") or
                                  (word.startswith("(") and not word.startswith("((")))):
            change += 1                     # a group, function, subshell, array or substitution
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


# A path to a shell script, read after quotes and backslashes are removed; a ``$ROOT/`` or
# ``${ROOT}/`` prefix, the repository root in every script, is dropped.
SCRIPT_MENTION = re.compile(r"(?:\$\{?ROOT\}?/)?([\w./+@%~,-]*\.sh)(?![\w.-])")


def mentioned_scripts(text: str) -> list[str]:
    """Every shell script a text names, in order, as a normalised repository-relative path.

    Every way of naming a script counts, in a quoted command string, a substitution, a
    subshell, a default value or glued to other text. A name that cannot be resolved to a
    path, such as ``${step}.sh``, ``{a,b}.sh``, ``$SCRIPT_DIR/a.sh`` or ``./a.sh`` after a
    cd, comes out as what it can be read as, which differs from every scanned path, so a test
    comparing them fails loudly rather than taking it for one already scanned.
    """
    plain = re.sub(r"[\\'\"]", "", text)
    plain = re.sub(r"\$\{\w+:?[-=+?]", " ", plain)       # a default value is a word of its own
    return [posixpath.normpath(path) for path in SCRIPT_MENTION.findall(plain)]


def called_scripts(lines: list[ShellLine]) -> set[str]:
    """The shell scripts a script may run: every one it names, as mentioned_scripts reads it.

    Here-document bodies count too, as one may be piped to a shell.
    """
    return {path for line in lines for path in mentioned_scripts(line.text)}
