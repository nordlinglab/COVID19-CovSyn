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


@dataclass
class _Frame:
    """One open context of the walker.

    Its kind, where it opened, its parenthesis or bracket nesting, and, in a command
    substitution, how many case statements are open in it.
    """

    kind: str
    start: int
    nesting: int = 0
    cases: int = 0
    command_start: bool = True
    in_pattern: bool = False
    expect_in: bool = False


@dataclass
class _Walk:
    """What the context walker found in a text.

    ``kinds`` holds, for each character, the context it stands in: ``top`` (plain code), a
    quote (``'``, ``"``, ``$'``, a backtick), the inside of ``$(`` (plain code too), of the
    arithmetic ``$((``, ``((`` and ``$[``, of ``${``, or ``#`` for a comment. ``groups`` maps
    the index where each ``$(``, ``$((``, ``((``, ``$[``, ``${`` or backtick group opens to the
    index of its last character; a process substitution ``<(`` or ``>(`` is recorded like a
    ``$(``. ``stack`` holds the contexts open at the end, the outermost ``top`` first, so a walk
    can go on from it. ``closers`` holds the index of each ``)`` that closes a substitution
    (``$(``, ``$((``, ``<(``), after which the word goes on, unlike after the ``))`` of an
    arithmetic command.
    """

    kinds: list[str]
    groups: dict[int, int]
    stack: list[_Frame]
    closers: set[int]

    @property
    def open(self) -> list[str]:
        """The contexts still open at the end of the text, innermost last."""
        return [frame.kind for frame in self.stack[1:]]


PLAIN = {"top", "$("}                   # commands: where << redirects and # comments
ARITHMETIC = {"$((", "((", "$["}
# Where $(, $((, ${ and $[ open: bash expands them in arithmetic too.
ACTIVE = PLAIN | {"${", '"'} | ARITHMETIC
GROUPS = {"$(", "$((", "((", "$[", "${", "`"}


# Words after which a command starts on the same line: reserved words, the brace of a group,
# and the openers whose condition is a command. block_change and _walk share them.
LEADERS = {"then", "else", "elif", "do", "!", "time", "{", "if", "while", "until"}
# The characters after which bash starts a new word, so that a # there starts a comment; a )
# that closes a substitution is no break, as the word goes on after it.
WORD_BREAKS = " \t\n;&|()<>"
WORD = re.compile(rf"[^{re.escape(WORD_BREAKS)}]*")


def _arithmetic_closes(text: str, start: int) -> bool:
    """Whether the ``$((`` at ``start`` closes with ``))``, so that bash reads arithmetic.

    bash reads ``$((cd x) | wc)``, whose first group closes with a single ``)``, as a command
    substitution starting with a subshell; quoted and escaped parentheses do not count. The
    look-ahead stops at the end of the line, so a
    walk line by line reads the same as a walk of the whole text; an expansion still open there
    is taken as arithmetic.
    """
    depth, index, quote = 0, start + 3, None
    while index < len(text) and text[index] != "\n":
        char = text[index]
        if char == "\\" and quote != "'":
            if text[index + 1:index + 2] == "\n":
                break                       # the line ends here; decide as at its end
            index += 2
            continue
        if quote:
            quote = None if char == quote else quote
        elif char in "'\"":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            if depth:
                depth -= 1
            else:
                # The first unmatched ) decides: )) is arithmetic, and so is one ending the
                # whole text, where bash waits for more; one before anything else, a newline
                # included, makes bash read a subshell (bash 5: $((1+2)\n) is one).
                return text[index + 1:index + 2] in {")", ""}
        index += 1
    return True


def _arithmetic_frame(stack: list[_Frame]) -> _Frame | None:
    """The arithmetic frame a ${...} sits in, through any number of ${...}, or None.

    bash counts every parenthesis up to the closing )) of an arithmetic expansion or command,
    those inside a nested ${...} included.
    """
    for frame in reversed(stack):
        if frame.kind in {"$((", "(("}:      # $[ counts brackets, not parentheses
            return frame
        if frame.kind != "${":
            return None
    return None


def _walk(text: str, stack: list[_Frame] | None = None, before: str = "\n") -> _Walk:
    """Walk a text as bash reads its quoting, going on from ``stack`` when one is given.

    A backslash escapes the next character outside single quotes, inside ``$'...'`` too. A ``$(``
    opens plain code again inside double quotes, ``${...}`` and arithmetic, where ``${``, ``$((``
    and ``$[`` open too, as bash expands them there; inside backticks it is text, as bash reads
    backtick content as a string. ``<(`` and ``>(`` open a group in plain code and in any
    ``${...}``, also one inside double quotes or arithmetic, where bash parses the group although it
    does not run it; directly inside double quotes they are text. ``$((``, ``((`` at the start of a
    word and ``$[`` open arithmetic, which counts the parentheses of a nested ``${...}`` too. In a
    command substitution a ``)`` that ends a case pattern does not close it; ``case`` and ``esac``
    count only where a command starts, and in a case's pattern position (opened by the ``in`` right
    after its subject and by ``;;``, ``;&`` or ``;;&``) only ``esac`` is reserved, not right after a
    pattern's optional ``(``. A ``#`` at the start of a word in plain code, not after an escaped
    blank, starts a comment to the end of its line. A character that opens or closes a context
    belongs to the outer one. ``before`` is the character that precedes the text, which decides
    whether it starts a word; frames taken over from ``stack`` record no group, as their start lies
    in another text.
    """
    stack = [_Frame(f.kind, -1, f.nesting, f.cases, f.command_start, f.in_pattern, f.expect_in)
             for f in stack] if stack else [_Frame("top", -1)]
    kinds, groups, closers, index, escaped_end = [], {}, set(), 0, -1

    def push(kind: str, width: int) -> None:
        nonlocal index
        kinds.extend([stack[-1].kind] * width)
        stack.append(_Frame(kind, index))
        index += width

    def pop(width: int) -> None:
        nonlocal index
        frame = stack.pop()
        if frame.kind in {"$(", "$(("}:     # the word goes on after a substitution's )
            closers.add(index + width - 1)
        if frame.kind in GROUPS and frame.start >= 0:
            groups[frame.start] = index + width - 1
        kinds.extend([stack[-1].kind] * width)
        index += width

    while index < len(text):
        frame = stack[-1]
        kind, char = frame.kind, text[index]
        previous = text[index - 1] if index else before
        word_start = previous in WORD_BREAKS and escaped_end != index and \
            (index - 1) not in closers
        if kind == "#":
            if char == "\n":
                stack.pop()
                kinds.append(stack[-1].kind)
            else:
                kinds.append("#")
            index += 1
        elif kind == "'":
            if char == "'":
                pop(1)
            else:
                kinds.append(kind)
                index += 1
        elif char == "\\":
            width = len(text[index:index + 2])
            kinds.extend([kind] * width)
            index += width
            escaped_end = index
        elif kind == "$'":
            if char == "'":
                pop(1)
            else:
                kinds.append(kind)
                index += 1
        elif (kind == '"' and char == '"') or (kind == "`" and char == "`") or \
                (kind == "${" and char == "}") or \
                (kind == "$[" and char == "]" and frame.nesting == 0) or \
                (kind == "$(" and char == ")" and frame.nesting == 0 and frame.cases <= 0):
            pop(1)
        elif kind in {"$((", "(("} and text.startswith("))", index) and frame.nesting == 0:
            pop(2)
        elif kind in ACTIVE and text.startswith("$((", index) and _arithmetic_closes(text, index):
            push("$((", 3)
        elif kind in ACTIVE and text.startswith("$(", index):
            push("$(", 2)
        elif kind in PLAIN | {"${"} and char in "<>" and text.startswith("(", index + 1) and \
                previous not in "<>":
            push("$(", 2)                   # a process substitution, parsed like $(
        elif kind in ACTIVE and text.startswith("${", index):
            push("${", 2)
        elif kind in ACTIVE and text.startswith("$[", index):
            push("$[", 2)
        elif kind in PLAIN and word_start and text.startswith("((", index) and \
                previous not in "<>":           # <(( is a process substitution
            push("((", 2)
        elif kind in PLAIN and word_start and char == "#":
            stack.append(_Frame("#", index))
            kinds.append("#")
            index += 1
        elif kind in PLAIN | {"${"} and text.startswith("$'", index):
            push("$'", 2)
        elif (kind in PLAIN | {"${"} and char in "'\"`") or (kind == '"' and char == "`"):
            push(char, 1)
        else:
            if kind == "$(" and word_start and not char.isspace():
                word = WORD.match(text, index).group()
                if frame.command_start and (word == "esac" or
                                            (word == "case" and not frame.in_pattern)):
                    # In pattern position only esac is reserved; a pattern may read case.
                    frame.cases += 1 if word == "case" else -1
                    frame.in_pattern, frame.expect_in = False, word == "case"
                    frame.command_start = False
                elif word == "in" and frame.expect_in:
                    frame.in_pattern, frame.expect_in = True, False   # the patterns follow
                    frame.command_start = True  # so that an empty case's esac is read
                else:
                    frame.command_start = word in LEADERS
            pattern_paren = kind == "$(" and char == "(" and frame.in_pattern and \
                frame.nesting == 0
            if kind == "$(" and (char in ";&\n" or (char == "(" and not pattern_paren) or
                                 (char == "|" and not frame.in_pattern)):
                frame.command_start = True
            elif pattern_paren:
                frame.command_start = False     # bash: no esac right after a pattern's (
            if kind == "$(" and char == ";" and text.startswith((";;", ";&"), index) and \
                    frame.cases > 0:
                frame.in_pattern = True         # after ;; ;& ;;& the next pattern or esac
            if kind == "$(" and char == ")" and frame.nesting == 0 and frame.cases > 0:
                frame.command_start = True      # a case pattern ends; its command follows
                frame.in_pattern = False
            elif pattern_paren:
                pass                            # the optional ( before a case pattern
            elif kind in {"$(", "$((", "(("} and char in "()":
                frame.nesting += 1 if char == "(" else -1
            elif kind == "$[" and char in "[]":
                frame.nesting += 1 if char == "[" else -1
            elif kind == "${" and char in "()" and (arithmetic := _arithmetic_frame(stack)):
                arithmetic.nesting += 1 if char == "(" else -1
            kinds.append(kind)
            index += 1
    while stack[-1].kind == "#":            # a comment ends with its line
        stack.pop()
    return _Walk(kinds, groups, stack, closers)


def _code(text: str, walk: _Walk) -> str:
    """A walked text without its comments."""
    return "".join(c for c, kind in zip(text, walk.kinds, strict=True) if kind != "#")


def _scan_quotes(text: str) -> tuple[str, str | None]:
    """A text without its comments, and the context still open at its end, as bash reads it.

    Any open quote, substitution, arithmetic or ``${`` continues the logical line, as in bash.
    """
    walk = _walk(text)
    return _code(text, walk), walk.open[-1] if walk.open else None


def strip_comment(line: str) -> str:
    """A line without its comment, as bash reads it."""
    return _scan_quotes(line)[0]


def _trailing_backslashes(text: str) -> int:
    return len(text) - len(text.rstrip("\\"))


def _fold(line: str, openers: tuple[str, ...]) -> str:
    """The line with each closed group that opens with one of ``openers`` as X.

    Groups come from one walk, which reads case patterns inside a substitution; a quoted opener
    is text and opens none, and a group left open is kept. Nested groups fold with the
    outermost.
    """
    walk, spans = _walk(line), []
    for start, end in sorted(walk.groups.items()):
        if not any(line.startswith(o, start) for o in openers):
            continue
        if spans and start <= spans[-1][1]:
            continue                        # inside a group already folded
        spans.append((start, end))
    for start, end in reversed(spans):
        line = line[:start] + "X" + line[end + 1:]
    return line


def _fold_arithmetic(line: str) -> str:
    """The line with each arithmetic expansion and command as X, ``$[ ]`` included."""
    return _fold(line, ("$((", "((", "$["))


def _fold_substitutions(line: str) -> str:
    """The line with each substitution as X: arithmetic, ``$( )``, ``<( )`` and backticks."""
    return _fold(line, ("$((", "((", "$[", "$(", "`", "<(", ">("))


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

    Plain code is outside quotes, ``${...}`` and arithmetic, or inside a ``$(...)`` wherever
    that stands; only there does bash take ``<<`` for a redirection.
    """
    return [kind in PLAIN for kind in _walk(line).kinds]


def _shell_word(text: str) -> str | None:
    """The first shell word of a text, quotes and backslashes removed; None if there is none.

    That is how bash reads a here-document delimiter; an unparsable word gives None too.
    """
    text = text.lstrip(" \t")
    kinds, end = _walk(text).kinds, 0
    while end < len(text) and not (kinds[end] == "top" and
                                   (text[end].isspace() or text[end] in ";&|<>()")):
        end += 1
    try:
        words = shlex.split(text[:end])
    except ValueError:
        return None
    return words[0] if words else None


def _heredoc_delimiters(line: str, kinds: list[str] | None = None) -> list[tuple[bool, str]]:
    """The here-documents a line opens: whether tabs are stripped, and the delimiter.

    ``<<`` opens one wherever bash takes it for a redirection: in plain code, inside a command
    substitution too, even within double quotes or ``${...}``, and in ``let x=1<<2`` as bash
    does (checked with bash 5); not inside quotes, ``${...}``, arithmetic or backticks.
    ``<<<`` is a here-string. The delimiter is the next shell word with its quotes and
    backslashes removed. ``kinds`` are the line's contexts when it continues a walk.
    """
    if "<<" not in line:
        return []
    kinds = kinds if kinds is not None else _walk(line).kinds
    found = []
    for match in re.finditer(r"(?<!<)<<(?!<)(-?)", line):
        if kinds[match.start()] in PLAIN:
            delimiter = _shell_word(line[match.end():])
            if delimiter:
                found.append((match.group(1) == "-", delimiter))
    return found


def shell_lines(text: str) -> list[ShellLine]:
    """A script's logical lines.

    A line still inside a quote, substitution, arithmetic or ``${`` at its end goes on with the
    next, walked from where it stopped; comments are dropped before continuation lines are
    joined (bash does not continue a comment), an odd run of trailing backslashes joins the
    next line into the same word, and blank lines are left out. A here-document body is kept
    as such, since it holds text, and follows its command line in the result. bash reads the
    body after the physical line that opens it when that line ends inside a command
    substitution, and after the logical line otherwise.
    """
    lines, buffer, stack, before = [], "", None, "\n"
    delimiters, pending, bodies = [], [], []
    for raw in text.splitlines():
        if delimiters:
            strip_tabs, word = delimiters[0]
            if (raw.lstrip("\t") if strip_tabs else raw) == word:
                delimiters.pop(0)
            else:
                bodies.append(ShellLine(raw, heredoc=True))
            if not delimiters and not buffer:
                lines += bodies
                bodies = []
            continue
        # The newline belongs to the line, as in bash: it ends a comment and starts a command.
        walk = _walk(raw + "\n", stack, before)
        code = _code(raw + "\n", walk)[:-1]
        pending += _heredoc_delimiters(raw, walk.kinds)
        # Inside '...' and $'...' a backslash-newline is text, which bash keeps.
        continues = _trailing_backslashes(code) % 2 and \
            not (walk.open and walk.open[-1] in {"'", "$'"})
        if continues:
            buffer, stack = buffer + code[:-1], walk.stack
            # The next line goes on the same word unless a blank or an operator, itself not
            # escaped, ends this one.
            last = code[-2] if len(code) > 1 else before
            escaped = len(code) > 1 and _trailing_backslashes(code[:-2]) % 2 == 1
            closes = len(code) - 2 in walk.closers      # a ) closing a group goes on the word
            before = last if last in WORD_BREAKS and not escaped and not closes else "x"
            continue
        if walk.open:
            buffer, stack, before = buffer + code + "\n", walk.stack, "\n"
            if walk.open[-1] == "$(":
                delimiters, pending = pending, []
            continue
        line, buffer, stack, before = buffer + code, "", None, "\n"
        if line.strip():
            lines.append(ShellLine(line))
        lines += bodies
        bodies = []
        delimiters, pending = pending, []
    if buffer.strip():
        lines.append(ShellLine(buffer))
    return lines + bodies


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
SCRIPT_MENTION = re.compile(
    r"(?:\$\{?ROOT\}?/)?((?:\$\{[^}]*\}|\$\w+|[\w./+@%~,-])*\.sh)(?![\w.-])")


def mentioned_scripts(text: str) -> list[str]:
    """Every shell script a text names, in order, as a normalised repository-relative path.

    Every way of naming a script counts, in a quoted command string, a substitution, a
    subshell, after an option or an assignment, or glued to other text; backtick content is
    read again, as bash does. A name that cannot be resolved to a path, such as
    ``${step}.sh``, ``{a,b}.sh``, ``$SCRIPT_DIR/a.sh``, ``${DIR}scripts/a.sh``,
    ``$(pwd)scripts/a.sh``, a default value ``${X:-scripts/a.sh}`` or ``./a.sh`` after a cd,
    comes out with its variable parts kept, which differs from every scanned path, so a test
    comparing them fails loudly rather than taking it for one already scanned.
    """
    plain = re.sub(r"[\\'\"]", "", text)
    walk, found = _walk(plain), []
    end_of_text = len(plain)
    # Parameter expansions and backtick groups as (start, end); an open one ends with the text.
    expansions = [(s, e) for s, e in walk.groups.items() if plain.startswith("${", s)]
    expansions += [(f.start, end_of_text) for f in walk.stack if f.kind == "${"]
    backticks = [(s, e) for s, e in walk.groups.items() if plain.startswith("`", s)]
    backticks += [(f.start, end_of_text) for f in walk.stack if f.kind == "`"]

    def enclosing(position: int) -> int | None:
        """The start of the outermost expansion holding a position, if any."""
        starts = [s for s, e in expansions if s < position <= e]
        return min(starts) if starts else None

    # Backtick content is read again, unless an expansion holds it, which is reported whole.
    for start, end in backticks:
        if enclosing(start) is None:
            found += mentioned_scripts(plain[start + 1:end])
    # A substitution that ends right before a path joins its word: its closing character mapped
    # to where it opens.
    closed_at = {end: start for start, end in walk.groups.items()
                 if plain.startswith(("$(", "<(", ">(", "`"), start)}
    inside_groups = {i for start, end in walk.groups.items()
                     if not plain.startswith("${", start) for i in range(start, end + 1)}
    for match in SCRIPT_MENTION.finditer(plain):
        opening = enclosing(match.start())
        if opening is not None:
            if re.fullmatch(r"\w+(?:%%?|##?)\.sh", plain[opening + 2:match.end()]):
                continue                    # ${f%.sh} removes an extension; it names nothing
            # A path in a default, alternative or pattern (${X:-bash scripts/a.sh}) is
            # reported with the expansion before it, so it differs from every scanned path.
            found.append(plain[opening:match.end()])
            continue
        if any(s < match.start() <= e for s, e in backticks):
            continue                        # read again above, as bash reads backticks
        # Text glued before the path, back to a word break, belongs to its word, a
        # substitution such as $(pwd) included.
        begin = match.start()
        while begin:
            if begin - 1 in closed_at:
                begin = closed_at[begin - 1]
            elif plain[begin - 1] not in WORD_BREAKS + "`":
                begin -= 1
            else:
                break
        glued = plain[begin:match.start()]
        # A leading NAME= or --option= whose value is the path alone is dropped; one whose value
        # is a : list, as PATH=$PATH:scripts/a.sh, stays, as does host: before a path. A colon
        # inside a substitution does not count.
        name = re.match(r"(?:[A-Za-z_]\w*|--?[\w-]+)=", glued)
        value = range(begin + (name.end() if name else 0), match.start())
        colon = any(plain[i] == ":" and i not in inside_groups for i in value)
        if name and not colon:
            glued = glued[name.end():]
        found.append(posixpath.normpath(glued + match.group(1)))
    return found


def called_scripts(lines: list[ShellLine]) -> set[str]:
    """The shell scripts a script may run: every one it names, as mentioned_scripts reads it.

    Here-document bodies count too, as one may be piped to a shell.
    """
    return {path for line in lines for path in mentioned_scripts(line.text)}
