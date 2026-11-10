# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The static environment scans of env_scan, which the B86 forwarding tests rely on."""

from __future__ import annotations

import shlex
from pathlib import Path

import env_scan as es
import pytest
from hypothesis import given
from hypothesis import strategies as st


def _module(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "module.py"
    path.write_text(source)
    return path


def _script(text: str) -> list[es.ShellLine]:
    return es.shell_lines(text)


# --- Python ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("source", "expected"), [
    ("import os\nos.environ['A']", {"A"}),
    ("import os\nos.environ.get('A')", {"A"}),
    ("import os\nos.environ.setdefault('A', '1')", {"A"}),
    ("import os\nos.environ.pop('A')", {"A"}),
    ("import os\nos.getenv('A')", {"A"}),
    ("import os\nos.getenv(key='A')", {"A"}),
    ("from os import getenv\ngetenv('A')", {"A"}),
    ("from os import environ\n'A' in environ", {"A"}),
    ("import os\n'A' not in os.environ", {"A"}),
    ("from os import environ\nenviron['A']", {"A"}),
    ("from os import getenv as g\ng('A')", {"A"}),
    ("from os import environ as e\ne.get('A')", {"A"}),
    ("import os\ne = os.environ\ne.get('A')", {"A"}),
    ("import os\ne: dict = os.environ\ne.get('A')", {"A"}),
    ("import os\n(e := os.environ).get('A')", {"A"}),
    ("import os\ndef f(x=os.environ):\n    return x.get('A')", {"A"}),
    ("import os\nb = a\na = os.environ\nb.get('A')", {"A"}),   # bound in either order
    ("import os as o\no.environ['A']", {"A"}),
    ("def f(env):\n    return env.get('A')", {"A"}),
    ("# os.environ['A']\nprint('os.getenv(B)')", set()),
    ("d = {}\nd.get('A')", set()),
    ("x = config.environ\nx.get('A')", set()),              # not the environment of os
    ("import os\ncfg = os.environ.copy()\ncfg['A']", {"A"}),
    ("import os\ne = dict(os.environ)\ne.get('A')", {"A"}),
    ("import os\n'A' in os.environ.keys()", {"A"}),
    ("import os\ne, f = os.environ, 1\ne.get('A')", {"A"}),
    ("import os\n[e, f] = [os.environ, 1]\ne.get('A')", {"A"}),
    ("import os\nx = {**os.environ}\nx['A']", {"A"}),
    ("import os\nx = dict(**os.environ)\nx['A']", {"A"}),
    ("import copy, os\nx = copy.copy(os.environ)\nx['A']", {"A"}),
    ("import os\nx = {**os.environ, 'B': '1'}\nx.get('A')", {"A"}),
    ("import os\nfrom copy import deepcopy\nx = deepcopy(os.environ)\nx['A']", {"A"}),
    ("import os\nimport copy as cp\nx = cp.copy(os.environ)\nx['A']", {"A"}),
    ("import sys\nsys.modules['os'].environ['A']", {"A"}),     # any .environ, for a read
    ("self.env.get('A')", {"A"}),
    ("def f(env):\n    e = env.copy()\n    return e['A']", {"A"}),
    ("def f(env):\n    e = dict(env)\n    return e.get('A')", {"A"}),
])
def test_environment_reads_are_found_in_code_only(tmp_path, source, expected):
    """Every spelling of an environment read is found, and nothing in comments or text."""
    assert es.environment_reads(_module(tmp_path, source)) == expected


@pytest.mark.parametrize(("source", "touches"), [
    ("import os\nworker_count(dict(os.environ))", True),
    ("import os\nenv = os.environ\nworker_count(env)", True),
    ("from os import getenv\ngetenv('A')", True),
    ("from os import environ as e\nworker_count(e)", True),
    ("from os import getenv as g\ng('A')", True),
    ("import os\nos.environb", True),
    ("import os\ngetattr(os, 'environ')", True),
    ("import os as o\ngetattr(o, 'environ')", True),
    ("import os\nvars(os)['environ']", True),
    ("x = 1\n'environ'", True),                  # a bare string statement, not a docstring
    ("def worker_count(env):\n    return env.get('A')", False),
    ("# os.environ\nprint('os.environ')", False),
    ("def f():\n    '''environ'''", False),
])
def test_touching_the_environment_is_found(tmp_path, source, touches):
    """Copies and aliases of the environment count; a mapping parameter or text does not."""
    assert es.mentions_environment(_module(tmp_path, source)) == touches


def test_the_import_closure_follows_relative_imports_and_packages(tmp_path):
    """A relative import and a package ``__init__.py`` are part of what a module runs."""
    package = tmp_path / "covsyn" / "calibration"
    package.mkdir(parents=True)
    (tmp_path / "covsyn" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("import os\nos.environ['B']")
    (package / "main.py").write_text("from .helper import f\n")
    (package / "helper.py").write_text("def f():\n    pass\n")
    closure = es.import_closure("covsyn.calibration.main", tmp_path / "covsyn")
    files = {p.relative_to(tmp_path).as_posix() for p in closure}
    assert files == {"covsyn/__init__.py", "covsyn/calibration/__init__.py",
                     "covsyn/calibration/main.py", "covsyn/calibration/helper.py"}
    assert any(es.mentions_environment(p) for p in closure)


# --- Shell ----------------------------------------------------------------------------------


@pytest.mark.parametrize(("line", "code"), [
    ("echo hi # note", "echo hi "),
    ("# whole line", ""),
    ("echo '#x' \"#y\" a#b", "echo '#x' \"#y\" a#b"),
    ("echo ${#x}", "echo ${#x}"),
    ("echo \\# x", "echo \\# x"),
    ("f; # note", "f; "),
])
def test_comments_are_stripped_as_bash_reads_them(line, code):
    """A # starts a comment only at the start of a word outside quotes."""
    assert es.strip_comment(line) == code


def test_a_comment_does_not_continue():
    """A comment ending in a backslash, whole-line or trailing, does not swallow the next line."""
    assert es.shell_lines("# note \\\necho ${A:-x}\n") == [es.ShellLine("echo ${A:-x}")]
    assert es.shell_lines("B=1  # see \\\necho $B\n")[1] == es.ShellLine("echo $B")


def test_a_quoted_string_spans_lines():
    """The lines inside a multi-line quoted string are text, not commands or comments."""
    lines = _script('MSG="x\nA=1\n"\necho $A\necho "a \\\n# ${X}"\n')
    assert es.shell_variables(lines)[1] == {"MSG"}
    assert "X" in es.shell_variables(lines)[0]


@pytest.mark.parametrize(("text", "continues"), [
    ("echo a\\\nB=1\n", True),          # one backslash: a continuation
    ("echo a\\\\\nB=1\n", False),     # two: an escaped backslash
    ("echo a\\\\\\\nB=1\n", True),  # three: escaped backslash, then a continuation
])
def test_an_odd_run_of_backslashes_continues(text, continues):
    """Only an odd run of trailing backslashes ends in an unescaped one."""
    assert (len(_script(text)) == 1) == continues


def test_continuation_lines_are_joined():
    """A NAME= argument on a continuation line belongs to the command it continues."""
    lines = _script("run x \\\n  A=1\necho $A\n")
    assert lines[0].text.startswith("run x")
    assert es.shell_variables(lines)[1] == set()


def test_heredoc_bodies_are_text():
    """A here-document holds text, so a NAME= in it assigns nothing; its $NAME still expands."""
    lines = _script("cat <<EOF\nA=1 $B\nEOF\necho $A\n")
    assert lines[1] == es.ShellLine("A=1 $B", heredoc=True)
    referenced, assigned, _ = es.shell_variables(lines)
    assert assigned == set()
    assert {"A", "B"} <= referenced


def test_nested_arithmetic_opens_no_block():
    """An arithmetic expansion with nested parentheses folds into one word."""
    assert es.block_change("x=$(( (1+2) ))") == 0


def test_a_case_pattern_starts_a_command():
    """An opener after a case pattern counts, so a block opened and closed there nets zero."""
    assert es.block_change("case x in a) if y; then z; fi;; esac") == 0


@pytest.mark.parametrize(("line", "words"), [
    ("A=$B", [["A=$B"]]),
    ("A=${B}x", [["A=${B}x"]]),
    ("A=$((1+2))", [["A=X"]]),
    ("A=$'x'", [["A=$x"]]),
    ("A=a,b", [["A=a,b"]]),
    ("A=1 2>/dev/null", [["A=1"]]),
    ("A=1 > out", [["A=1"]]),
    ("A=\"$(cd \"$(dirname x)\" && pwd)\"", [["A=X"]]),
    ("A=1; echo hi", [["A=1"], ["echo", "hi"]]),
    ('A=$(echo ")")', [["A=X"]]),                                  # a quoted ) is text
    ("x=$( echo $(a)", [["x=$(", "echo", "X"]]),                   # the closed one still folds
    ('A=$(echo "a\\")b")', [["A=X"]]),                           # an escaped quote
    ('A=$(echo "$(echo ")")")', [["A=X"]]),                       # quotes nested in $( )
    ("a && b || c | d & e", [["a"], ["b"], ["c"], ["d"], ["e"]]),
])
def test_commands_are_split_into_words(line, words):
    """Words split at whitespace and command separators.

    Substitutions are folded into one word and redirections dropped.
    """
    assert es.commands(line) == words


@pytest.mark.parametrize(("lines", "assigned"), [
    (["A=1", "echo $A"], {"A"}),
    (["A=$B", "echo $A"], {"A"}),
    (["A=1; echo hi", "echo $A"], {"A"}),
    (["A=1 cmd", "echo $A"], set()),                         # sets A for cmd only
    (["A=$B cmd", "echo $A"], set()),
    (["echo $A", "A=1"], set()),                             # read before it is assigned
    (["if x; then", "  A=1", "fi", "echo $A"], set()),       # may not run
    (["if x; then", "A=1", "fi", "echo $A"], set()),
    (["f() {", "A=1", "}", "echo $A"], set()),               # runs only if f is called
    (["for i in 1; do", "A=1", "done", "echo $A"], set()),
    (["if x; then y; fi", "A=1", "echo $A"], {"A"}),         # the block closed on its line
    (["for A in 1; do :; done"], set()),
    (["if a; then", ":", "else if b; then", ":", "fi", "A=1", "fi", "echo $A"], set()),
    (["if x; then", 'echo "}"', "A=1", "fi", "echo $A"], set()),   # a quoted brace is text
    (["(", "A=1", ")", "echo $A"], set()),                        # a subshell
    (["cat <<\\EOF", "A=1", "EOF", "echo $A"], set()),           # a backslash-quoted delimiter
    (["x=$((1<<2))", "A=1", "echo $A"], {"A"}),                   # a shift, no here-document
    (["cat <<< word", "A=1", "echo $A"], {"A"}),                  # a here-string
    (["(( x = 1 << 2 ))", "A=1", "echo $A"], {"A"}),              # a shift in (( ))
    (["cat<<EOF", "A=1", "EOF", "echo $A"], set()),               # no space before EOF
    (["if x; then", "ARR=(", " a", ")", "A=1", "fi", "echo $A"], set()),
    (["if x; then", "Y=$(", " a", ")", "A=1", "fi", "echo $A"], set()),
    (["f () {", ":", "}", "A=1", "echo $A"], {"A"}),              # one function body
    # ANSI-C quoting no longer swallows the rest of the script; shlex cannot parse the value,
    # so X itself is not taken as assigned, which can only report more reads.
    (["X=$'it\\'s'", "A=1", "echo $A"], {"A"}),
    (["echo $'say \"hi\"'", "A=1", "echo $A"], {"A"}),          # " inside $'...' is text
    (["echo $'a #b'", "A=1", "echo $A"], {"A"}),                  # so is #
    (["func(){", "A=1", "}", "echo $A"], set()),                  # a function body, no spaces
    (['echo x"<<"EOF', "A=1", "echo $A"], {"A"}),                 # a quoted <<, no heredoc
    (['cat x"<<"<<EOF', "A=1", "EOF", "echo $A"], set()),         # the unquoted << counts
    (["cat <<EOF>out", "A=1", "EOF", "B=1", "echo $B"], {"B"}),    # a glued redirection
    (["cat <<A<<B", "x", "A", "y", "B", "C=1", "echo $C"], {"C"}),  # two here-documents
    (["x=$(cat <<EOF)", "A=1", "EOF", "echo $A"], set()),         # inside a substitution
])
def test_only_assignments_that_always_run_before_use_count(lines, assigned):
    """A name is the script's own only when assigned outside every block before its first use."""
    assert es.shell_variables(es.shell_lines("\n".join(lines)))[1] == assigned


@pytest.mark.parametrize(("line", "names"), [
    ("echo ${#A} ${!B} $((C+1))", {"A", "B", "C"}),
    ("(( D > 0 ))", {"D"}),
    ("echo $E ${F:-x}", {"E", "F"}),
])
def test_every_form_of_reading_a_variable_is_a_reference(line, names):
    """Length, indirection and arithmetic read a variable too."""
    assert es.references(line) == names


@pytest.mark.parametrize(("line", "called"), [
    ("bash scripts/a.sh", {"a.sh"}),
    ("sh scripts/a.sh", {"a.sh"}),
    ("source scripts/a.sh", {"a.sh"}),
    (". scripts/a.sh", {"a.sh"}),
    ("./scripts/a.sh", {"a.sh"}),
    ("scripts/a.sh x", {"a.sh"}),
    ("bash ./scripts/a.sh", {"a.sh"}),
    ('bash "$ROOT/scripts/a.sh"', {"a.sh"}),
    ("P=1 bash scripts/a.sh >> log", {"a.sh"}),
    ('bash "$ROOT"/scripts/a.sh', {"a.sh"}),
    ('bash "${REPO_ROOT}/scripts/a.sh"', {"a.sh"}),
    ("echo done", set()),
    ('echo "run scripts/a.sh later"', {"a.sh"}),     # named in text: reported, to be safe
    ("(bash scripts/a.sh)", {"a.sh"}),
    ("bash scripts/a.sh>log", {"a.sh"}),
    ("out=$(bash scripts/a.sh)", {"a.sh"}),
    ("x=`bash scripts/a.sh`", {"a.sh"}),
    ('tmux new-session -d -s x "bash scripts/a.sh"', {"a.sh"}),
    ("echo myscripts/a.sh", set()),
    ("true&&scripts/a.sh", {"a.sh"}),
    ("x;scripts/a.sh", {"a.sh"}),
    ("a|scripts/a.sh", {"a.sh"}),
    ("${X:-scripts/a.sh}", {"a.sh"}),
    ("echo scripts/a.sh.bak", set()),
    ("echo my-scripts/a.sh", set()),
    ("bash -x scripts/a.sh", {"a.sh"}),
    ("bash -o pipefail scripts/a.sh", {"a.sh"}),
    ("if bash scripts/a.sh; then :; fi", {"a.sh"}),
    ("do bash scripts/a.sh", {"a.sh"}),
    ("( bash scripts/a.sh )", {"a.sh"}),
    ("time bash scripts/a.sh", {"a.sh"}),
    ("nohup bash scripts/a.sh &", {"a.sh"}),
    ("tmux new-session -d -s x bash scripts/a.sh", {"a.sh"}),
    ("echo scripts/a.sh", {"a.sh"}),                 # a word of its own: reported, to be safe
])
def test_script_calls_are_found_however_written(line, called):
    """Every way of running a script under scripts/ is a call."""
    assert es.called_scripts([es.ShellLine(line)]) == called


# --- Properties -----------------------------------------------------------------------------

LINE_TEXT = st.text(st.characters(blacklist_characters="\n\r\x00", codec="utf-8"))
PLAIN_WORD = st.text(st.characters(whitelist_categories=("L", "N"), whitelist_characters="_./"),
                     min_size=1)


@given(LINE_TEXT)
def test_any_quoted_value_is_one_assignment(value):
    """NAME= followed by any one quoted word assigns NAME, and nothing else on the line."""
    assert es.assignment("A=" + shlex.quote(value)) == "A"


@given(LINE_TEXT, PLAIN_WORD)
def test_a_value_followed_by_a_command_is_no_assignment(value, command):
    """NAME=value followed by a command word only sets NAME for that command."""
    assert es.assignment(f"A={shlex.quote(value)} {command}") is None


@given(LINE_TEXT, LINE_TEXT)
def test_a_quoted_word_never_holds_a_comment(word, comment):
    """A # inside a quoted word is text; one after it, at a word start, starts the comment."""
    code = f"echo {shlex.quote(word)} "
    assert es.strip_comment(code) == code
    assert es.strip_comment(code + "# " + comment) == code


@given(st.lists(PLAIN_WORD, min_size=1, max_size=5))
def test_plain_words_survive_the_split(words):
    """Words without quotes or shell syntax come back as they went in."""
    assert es.commands(" ".join(words)) == [words]


@given(PLAIN_WORD)
def test_a_called_script_is_found_with_any_name(name):
    """A script is found under its own name, whatever characters the name uses."""
    script = f"{name.replace('/', '_')}.sh"
    assert es.called_scripts([es.ShellLine(f"bash scripts/{script}")]) == {script}


@given(LINE_TEXT)
def test_a_path_inside_quoted_text_is_reported(text):
    """A script named inside quoted text may be a command string, so it is reported."""
    assert "a.sh" in es.called_scripts([es.ShellLine(f"run {shlex.quote(text + ' scripts/a.sh')}")])


@given(st.sampled_from(["", "bash ", "(bash ", "x=$(bash ", "`bash ", "if bash ", '"bash ',
                        "env P=1 bash ", "nohup "]),
       st.sampled_from(["", ")", ">log", " >> log", "`", '"', "; :", " || exit 1", "&"]))
def test_a_script_is_found_in_any_structure(lead, trail):
    """However a call is wrapped, quoted or redirected, the script is found."""
    assert es.called_scripts([es.ShellLine(f"{lead}scripts/a.sh{trail}")]) == {"a.sh"}
