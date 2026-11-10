# Copyright 2026 Lee Cheng Jui <rexlee871221@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The static environment scans of env_scan, which the B86 forwarding tests rely on."""

from __future__ import annotations

import posixpath
import shlex
from pathlib import Path

import env_scan as es
import pytest
from hypothesis import assume, given
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
    ("echo a\\ #b", "echo a\\ #b"),                # an escaped blank keeps # in the word
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


def test_a_case_statement_in_a_substitution_folds_whole():
    """The unpaired ) of a case pattern does not end its command substitution."""
    line = "A=$(case $x in a) echo 1;; esac)"
    assert es.commands(line) == [["A=X"]]
    assert es.assignment(line) == "A"


def test_nested_arithmetic_opens_no_block():
    """An arithmetic expansion with nested parentheses folds into one word."""
    assert es.block_change("x=$(( (1+2) ))") == 0


def test_a_heredoc_body_follows_its_command():
    """The body of a here-document opened in a multi-line substitution comes after the line."""
    lines = es.shell_lines("X=$(cat <<EOF\nhello\nEOF\n)\n")
    assert [line.heredoc for line in lines] == [False, True]


def test_a_backslash_newline_in_ansi_c_quotes_is_kept():
    """A backslash and a newline inside $'...' stay, as in bash, so the line is not joined."""
    assert es.shell_lines("echo $'a\\\nb'\n") == [es.ShellLine("echo $'a\\\nb'")]


@pytest.mark.parametrize("text", ["echo $(echo a)#$Z\n", "A=$(echo a)\\\n#$Z\n",
                                  "echo $((1+2))#$Z\n", "cat <(echo a)#$Z\n"])
def test_a_hash_after_a_closed_group_stays_in_the_word(text):
    """The word goes on after the ) of a substitution, as in bash, so # there is no comment."""
    assert "Z" in es.shell_variables(es.shell_lines(text))[0]


def test_scan_reads_a_comment_after_an_operator_continuation():
    """After a semicolon and a continuation the next line starts a word, so # comments."""
    assert "Z" not in es.shell_variables(es.shell_lines("A=1;\\\n#b $Z\n"))[0]


def test_the_arithmetic_look_ahead_stops_at_an_escaped_newline():
    """An escaped newline ends the line for the look-ahead too."""
    assert es._arithmetic_closes("A=$((a \\\n) b)", 2)


@pytest.mark.parametrize("text", [
    "cat <(echo hi)",
    "echo ${X:-<(echo hi)}",
    '"${X:-<(echo hi)}"',                             # parsed as a group, though not run
    '"${A:-${B:-<(echo hi)}}"',
    "echo $(( ${X:-<(echo hi)} ))",
    "(( ${X:-<(echo hi)} ))",
    "echo $[ ${X:-<(echo hi)} ]",
])
def test_a_process_substitution_in_an_expansion_is_parsed_as_a_group(text):
    """A <( in plain code or in any ${...} is a bracketed group to bash."""
    assert text.index("<") in es._walk(text).groups


def test_a_process_substitution_directly_in_double_quotes_is_text():
    """Inside double quotes, outside any ${...}, <( is plain text to bash."""
    text = '"<(echo hi)"'
    assert text.index("<") not in es._walk(text).groups


def test_a_continued_walk_records_no_foreign_group():
    """A group opened in an earlier text is not recorded with indices from two texts."""
    first = es._walk("$(\n")
    assert es._walk("echo)", first.stack).groups == {}


def test_a_mention_of_case_does_not_stop_folding():
    """Only a case statement keeps a substitution unfolded, not the word case."""
    assert es._fold_substitutions("echo $(echo case) x") == "echo X x"


@pytest.mark.parametrize("line", [
    "A=$(if case $x in a) echo 1;; esac; then :; fi)",
    "A=$(while case $x in a) false;; esac; do :; done)",
    "A=$(case $x in a|esac) true;; esac)",           # esac as a pattern
    'A=$((echo "))") | wc)',                          # a quoted )) in a subshell
    "A=$(case $x in a) for i in 1; do echo | case $y in b) :;; esac; done;; c) :;; esac)",
    "A=$(case $x in case) :;; esac)",                 # a pattern spelled case
    "A=$(case $x in (a) case $y in b) :;; esac;; c) :;; esac)",
    "A=$(case $x in (esac) :;; esac)",               # bash: esac after ( is a pattern
    "A=$(case $x in a) :;& case) :;; esac)",          # a pattern after ;& reads case
])
def test_case_and_subshell_substitutions_fold_whole(line):
    """Each of these is one command substitution assigned to A."""
    assert es.commands(line) == [["A=X"]]


def test_arithmetic_starting_with_a_parenthesis_folds_whole():
    """$(((1+2)*3)) is one arithmetic expansion, not $( followed by (( )."""
    assert es._fold_arithmetic("echo $(((1+2)*3)) x") == "echo X x"


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
    ("A=$(echo $'\\'')", [["A=X"]]),                              # ANSI-C quoting
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
    (['echo x"a <<EOF"', "B=1", "echo $B"], {"B"}),               # << quoted mid-word
    (['echo ${x:-"a <<b"}', "B=1", "echo $B"], {"B"}),
    (["echo ${x:-a <<b}", "B=1", "echo $B"], {"B"}),              # bash: no here-document
    (["cat <<E\\OF", "A=1", "EOF", "B=1", "echo $B"], {"B"}),     # delimiter EOF
    (["cat <<'EOF'x", "A=1", "EOF", "EOFx", "echo $A"], set()),    # delimiter EOFx
    (['cat <<"E F"', "A=1", "E F", "B=1", "echo $B"], {"B"}),
    (["cat <<@@", "A=1", "@@", "B=1", "echo $B"], {"B"}),
    (["let x=1<<2", "A=1", "2", "B=1", "echo $B"], {"B"}),          # bash: a here-document
    (["y=$((1<<2)); case $y in 4) :;; esac", "A=1", "echo $A"], {"A"}),  # case later on
    (['echo "$(cat <<EOF)"', "A=1", "EOF", "echo $A"], set()),    # bash: a here-document
    (["echo ${x:-$(cat <<EOF)}", "A=1", "EOF", "echo $A"], set()),
    (["echo $[1<<2]", "A=1", "echo $A"], {"A"}),                  # legacy arithmetic
    (["echo $[a[1]<<2]", "A=1", "echo $A"], {"A"}),               # nested brackets
    (["echo '$['; cat <<EOF; echo ']'", "A=1", "EOF", "echo $A"], set()),  # a quoted $[
    (["echo '$(( ' ; cat <<EOF ; echo ' ))'", "A=1", "EOF", "echo $A"], set()),
    (["x=`echo $(cat <<EOF)`", "A=1", "echo $A"], {"A"}),         # bash: text in backticks
    (['X="$(', "echo hi", ')"', "A=1", 'echo "$A"'], {"X", "A"}),  # a $( open in quotes
    (['echo "${x:-a', 'b}"', "A=1", "echo $A"], {"A"}),            # a ${ open in quotes
    (["echo hi # don't", "A=1", "echo $A"], {"A"}),               # a quote in a comment
    (["X=$(cat <<EOF", "hello", "don't", "EOF", ")", "A=1", "echo $A"], {"X", "A"}),
    (['X="$(cat <<EOF', "hello", "EOF", ')"', "A=1", "echo $A"], {"X", "A"}),
    (["echo '`' && while read x; do echo '`'", "A=1", "done", "echo $A"], set()),
    (["X=$(", "case $y in", "a) echo;;", "esac", ")", "A=1", "echo $A"], {"X", "A"}),
    (["X=$(echo case)", "A=1", "echo $A"], {"X", "A"}),          # case as an argument
    (["X=$(grep -w esac f)", "A=1", "echo $A"], {"X", "A"}),
    (["X=$(case $y in (a) echo;; esac)", "A=1", "echo $A"], {"X", "A"}),
    (["X=$(case $y in a) (cd /; ls);; esac)", "A=1", "echo $A"], {"X", "A"}),
    (["cat <<EOF; echo \"a", 'b"', "body", "EOF", "A=1", "echo $A"], {"A"}),  # body after
    (["X=$((cd /tmp && ls) | wc -l)", "A=1", "echo $A"], {"X", "A"}),  # a subshell, no arithmetic
    (["echo a\\", "#b", "A=1", "echo $A"], {"A"}),                 # #b joins the word a
    (["echo a\\ \\", "#b $Z", "A=1", "echo $A"], {"A"}),          # after an escaped blank
    (["X=$(", "echo a", "case $y in", "a) echo;;", "esac", ")", "A=1", "echo $A"],
     {"X", "A"}),                                                 # case after a command
    (["B=1;\\", "#b $Z", "echo $B"], {"B"}),                       # # after ; is a comment
    (["cmd >#; A=1", "echo $A"], set()),                         # bash: # after > comments
    (["echo $(echo a)#$Z", "A=1", "echo $A"], {"A"}),             # ) of $( ) is no break
    (["cat <((echo p) | cat)", "A=1", "echo $A"], {"A"}),         # process substitution
    (["((1))#it's", "A=1", "echo $A $Q"], {"A"}),                 # # after (( )) comments
    (["echo ${X:-<(echo })}#$Z", "A=1", "echo $A"], {"A"}),       # <( inside ${
    (["(( ${X:-<(echo <<E)} ))", "A=1", "echo $A"], set()),       # bash parses <( and <<E
    (['echo "${C:-${B:-<(#)}}" $Z', "A=1", "echo $A"], set()),    # bash: # comments in <(
    (['echo "${A:-${B:-<(cat <<E)}}"', "C=1", "E", "echo $C"], set()),  # a here-document
    (["echo $(( ${X:-<(echo })} ))", "A=1", "echo $A"], {"A"}),    # } inside the group
    (['echo "${X:-<(echo })"', "A=1", 'echo "$A"'], set()),         # bash: an open quote
    (["echo $(( ${X:-(} ))", "A=1", "echo $A"], set()),             # bash: ( counts
    (["(( $(cat <<E) ))", "A=1", "E", "B=1", "echo $B"], {"B"}),   # bash: a here-document
    (["A=$(case a in a) echo hi;; esac>/dev/null)", "B=1", "echo $B"], {"A", "B"}),
    (["X=$(echo a\\", "#b)", "A=1", "echo $A"], {"X", "A"}),       # a#b, one word
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


A = "scripts/a.sh"


@pytest.mark.parametrize(("line", "called"), [
    ("bash scripts/a.sh", {A}),
    ("sh scripts/a.sh", {A}),
    ("source scripts/a.sh", {A}),
    (". scripts/a.sh", {A}),
    ("./scripts/a.sh", {A}),
    ("scripts/a.sh x", {A}),
    ("bash ./scripts/a.sh", {A}),
    ('bash "$ROOT/scripts/a.sh"', {A}),
    ('bash "$ROOT"/scripts/a.sh', {A}),
    ('bash "${ROOT}/scripts/a.sh"', {A}),
    ("P=1 bash scripts/a.sh >> log", {A}),
    ('echo "run scripts/a.sh later"', {A}),           # named in text: reported, to be safe
    ("(bash scripts/a.sh)", {A}),
    ("bash scripts/a.sh>log", {A}),
    ("out=$(bash scripts/a.sh)", {A}),
    ("x=`bash scripts/a.sh`", {A}),
    ('tmux new-session -d -s x "bash scripts/a.sh"', {A}),
    ("bash scripts//a.sh scripts/./b.sh", {A, "scripts/b.sh"}),
    ('bash scripts/"a".sh', {A}),
    ("bash scripts/a\\.sh", {A}),
    ("true&&scripts/a.sh", {A}),
    ("x;scripts/a.sh", {A}),
    ("a|scripts/a.sh", {A}),
    ("${X:-scripts/a.sh}", {"${X:-scripts/a.sh"}),   # a default value: reported, differs
    ("bash ${X:=scripts/a.sh}", {"${X:=scripts/a.sh"}),
    ("bash ${X#scripts/a.sh}", {"${X#scripts/a.sh"}),
    ("bash --file=scripts/a.sh", {A}),                # after an option or an assignment
    ("S=scripts/a.sh", {A}),
    ("bash -o pipefail scripts/a.sh", {A}),
    ("if bash scripts/a.sh; then :; fi", {A}),
    ("nohup bash scripts/a.sh &", {A}),
    ("echo done", set()),
    ("echo a.shx scripts/a.sh.bak", set()),
    # Names that are no scanned path come out as something else, so a comparison fails:
    ("echo myscripts/a.sh", {"myscripts/a.sh"}),
    ("bash scripts/sub/a.sh", {"scripts/sub/a.sh"}),
    ("bash benchmarks/a.sh", {"benchmarks/a.sh"}),
    ('bash "$SCRIPT_DIR/a.sh"', {"$SCRIPT_DIR/a.sh"}),
    ('bash "${DIR}scripts/a.sh"', {"${DIR}scripts/a.sh"}),
    ("bash ${SUB:+sub/}scripts/a.sh", {"${SUB:+sub/}scripts/a.sh"}),
    ("cd scripts && bash ./a.sh", {"a.sh"}),
    ('bash "scripts/${step}.sh"', {"scripts/${step}.sh"}),
    ("bash scripts/{a,b}.sh", {"scripts/{a,b}.sh"}),
    ("bash scripts/a.sh,b.sh", {"scripts/a.sh,b.sh"}),
    ("bash scripts/a=b.sh", {"scripts/a=b.sh"}),      # the whole word, never its tail
    ("n=${f%.sh}", set()),                            # an extension removed, no script
    ("${X/old/scripts/a.sh}", {"${X/old/scripts/a.sh"}),   # any other expansion: loud
    ("${CMD:-scripts/run.sh --flag}", {"${CMD:-scripts/run.sh"}),
    ("${CMD:-bash scripts/a.sh}", {"${CMD:-bash scripts/a.sh"}),
    ("${X:-${Y} scripts/a.sh}", {"${X:-${Y} scripts/a.sh"}),   # the outermost expansion
    ("${X:-$(bash scripts/a.sh)}", {"${X:-$(bash scripts/a.sh"}),
    ("bash $(cd x; pwd)scripts/a.sh", {"$(cd x; pwd)scripts/a.sh"}),  # a built path
    ("echo $((${N}+1)); bash scripts/a.sh", {A}),     # ${ inside arithmetic is closed
    ("echo `echo ${X}` ; bash scripts/a.sh", {A}),
    ("echo $[${N}]; bash scripts/a.sh", {A}),
    ("echo `bash $(pwd)scripts/a.sh`", {"$(pwd)scripts/a.sh"}),  # backticks read again
    ("S=$(pwd)/scripts/a.sh", {"$(pwd)/scripts/a.sh"}),
    ("--file=$(pwd)scripts/a.sh", {"$(pwd)scripts/a.sh"}),
    ("${X:-`bash scripts/a.sh`}", {"${X:-`bash scripts/a.sh"}),
    ("echo ${X:-$(echo `bash scripts/a.sh`)}", {"${X:-$(echo `bash scripts/a.sh"}),
    ("echo $(( ${X:-$(bash scripts/a.sh)} ))", {"${X:-$(bash scripts/a.sh"}),
    ("(( ${X:-$(bash scripts/a.sh)} ))", {"${X:-$(bash scripts/a.sh"}),
    ("bash ${X%`echo .sh`}scripts/a.sh", {"${X%`echo .sh`}scripts/a.sh"}),
    ("echo `bash $(pwd)scripts/a.sh", {"$(pwd)scripts/a.sh"}),   # an open backtick
    ("host:`pwd`/scripts/a.sh", {"host:`pwd`/scripts/a.sh"}),
    ("PATH=$PATH:scripts/a.sh", {"PATH=$PATH:scripts/a.sh"}),
    ("S=$(echo a:b)/scripts/a.sh", {"$(echo a:b)/scripts/a.sh"}),   # a : inside $( )
    ("S=<(echo a:b)/scripts/a.sh", {"<(echo a:b)/scripts/a.sh"}),
    ("${X:-scripts/a-.sh}", {"${X:-scripts/a-.sh"}),
])
def test_scripts_are_found_however_named(line, called):
    """Every mention of a shell script counts, as a normalised repository-relative path."""
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
    """A script is found under its own path, whatever characters its name uses."""
    script = f"scripts/{name.replace('/', '_')}.sh"
    assert es.called_scripts([es.ShellLine(f"bash {script}")]) == {posixpath.normpath(script)}


@given(LINE_TEXT)
def test_a_path_inside_quoted_text_is_reported(text):
    """A script named inside quoted text may be a command string, so it is reported."""
    line = f"run {shlex.quote(text + ' scripts/a.sh')}"
    assert A in es.called_scripts([es.ShellLine(line)])


@given(st.sampled_from(["", "bash ", "(bash ", "x=$(bash ", "`bash ", "if bash ", '"bash ',
                        "env P=1 bash ", "nohup "]),
       st.sampled_from(["", ")", ">log", " >> log", "`", '"', "; :", " || exit 1", "&"]))
def test_a_script_is_found_in_any_structure(lead, trail):
    """However a call is wrapped, quoted or redirected, the script is found."""
    assert es.called_scripts([es.ShellLine(f"{lead}scripts/a.sh{trail}")]) == {A}


@given(LINE_TEXT)
def test_quoted_text_is_never_plain(text):
    """No character inside a single-quoted word is plain code, where << could redirect."""
    assume("'" not in text)            # one single-quoted word, no '"'"' splices
    line = f"echo '{text}'"
    assert not any(es._plain_positions(line)[len("echo '"):-1])


@given(LINE_TEXT, st.sampled_from(["", " rest", ";x", ">out", "<<B"]))
def test_a_quoted_delimiter_comes_back_whole(word, trail):
    """The shell word after << is read back with its quotes removed, up to what follows."""
    assume(word and word.strip(" \t") == word)
    assert es._shell_word(shlex.quote(word) + trail) == word


@given(LINE_TEXT)
def test_a_quoted_word_leaves_nothing_open(text):
    """Any text quoted for the shell opens no context that outlasts it."""
    code, open_context = es._scan_quotes(f"echo {shlex.quote(text)}")
    assert open_context is None
    assert code == f"echo {shlex.quote(text)}"


# A small grammar of balanced bash text: plain words, quotes, substitutions, arithmetic,
# parameter expansions and backticks, nested at random.
WORD = st.text(st.sampled_from("abc xyz019_"), min_size=1, max_size=6)
BALANCED = st.recursive(
    WORD,
    lambda inner: st.one_of(
        st.builds(lambda w: f"'{w}'", WORD),
        st.builds(lambda w: f'"{w}"', inner),
        st.builds(lambda w: f"$({w})", inner),
        st.builds(lambda w: f"$(( {w} ))", st.sampled_from(["1+2", "(1+2)*3", "x<<2"])),
        st.builds(lambda w: f"${{X:-{w}}}", inner),
        st.builds(lambda w: f"`{w}`", WORD),
        st.builds(lambda a, b: f"{a} {b}", inner, inner),
        st.builds(lambda w: f"$(case x in a) {w};; esac)", inner),
        st.builds(lambda w: f"$(case x in (a) {w};; esac)", inner),
        st.builds(lambda w: f"( {w} )", inner),
        st.builds(lambda w: f"a\\ {w}", inner),
        st.builds(lambda w: f"cat <({w})", inner),
        st.builds(lambda w: f"${{X:-<({w})}}", inner),
    ),
    max_leaves=8,
)


@given(BALANCED)
def test_balanced_text_leaves_nothing_open(text):
    """Text built from balanced constructs opens nothing that outlasts it.

    Every group the walker records starts with an opener and ends with its closer.
    """
    walk = es._walk(text)
    assert walk.open == []
    closers = {"$((": "))", "((": "))", "$(": ")", "${": "}", "$[": "]", "`": "`",
               "<(": ")", ">(": ")"}
    for start, end in walk.groups.items():
        opener = next(o for o in closers if text.startswith(o, start))
        if opener == "$((" and not es._arithmetic_closes(text, start):
            opener = "$("                   # a $( that starts with a subshell
        assert text[: end + 1].endswith(closers[opener])


@given(BALANCED, st.data())
def test_walking_line_by_line_equals_walking_whole(text, data):
    """A walk that goes on from the previous line's stack reads each line as a whole walk.

    The balanced text is cut at newlines put anywhere in it, also inside open contexts.
    """
    cuts = sorted(data.draw(st.sets(st.integers(0, len(text)), max_size=3)))
    parts = [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)], strict=True)]
    joined = "\n".join(parts)
    whole = es._walk(joined).kinds
    stack, kinds = None, []
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        walk = es._walk(part + ("" if last else "\n"), stack)
        kinds += walk.kinds
        stack = walk.stack
    assert kinds == whole


ARITHMETIC = st.recursive(st.sampled_from(["1", "x", "2"]),
                          lambda e: st.builds(lambda a, op, b: f"({a}{op}{b})", e,
                                              st.sampled_from(["+", "*", "<<"]), e),
                          max_leaves=5)


@given(ARITHMETIC, st.sampled_from(["", "x"]))
def test_arithmetic_is_told_from_a_subshell(expression, word):
    """$(( e )) closes with )) and is arithmetic; $((e) word) closes with ) and is not."""
    assert es._arithmetic_closes(f"$(( {expression} ))", 0)
    assert not es._arithmetic_closes(f"$(({expression}) {word})", 0)
