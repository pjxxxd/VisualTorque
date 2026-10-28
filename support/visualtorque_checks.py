"""File checks and notebook display utilities. No model or experiment dependencies."""
import functools
import hashlib
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path):
    return 'sha256:' + file_hash(path)


def sha256_file_by_name(file_path):
    return sha256_file(file_path)


sha = sha256_file
file_sha = sha256_file


def bytes_sha(data):
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def normalize_sha(value):
    return str(value).removeprefix('sha256:')


_HASH = re.compile(r'(?<![A-Za-z0-9_])(?:[0-9a-fA-F]{64}|[0-9a-fA-F]{40})(?![A-Za-z0-9_])')
_TEXT_MIMES = ('text/plain', 'text/html', 'text/markdown')


def compact_text(text, hashes=None):
    """Shorten displayed hashes, optionally keeping a reversible lookup."""
    def replace(match):
        value = match.group()
        short = value[:10] + '…' + value[-6:]
        if hashes is not None:
            if short in hashes and hashes[short] != value:
                return value
            hashes[short] = value
        return short
    return _HASH.sub(replace, text)


class _VisibleText(HTMLParser):
    def __init__(self, text, hashes):
        super().__init__(convert_charrefs=False)
        self.text, self.hashes, self.changes = text, hashes, []
        self.offsets = [0]
        self.offsets.extend(m.end() for m in re.finditer('\n', text))
        self.literal = None

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.literal = tag

    def handle_endtag(self, tag):
        if tag == self.literal:
            self.literal = None

    def handle_data(self, data):
        if self.literal:
            return
        changed = compact_text(data, self.hashes)
        if changed != data:
            line, column = self.getpos()
            start = self.offsets[line - 1] + column
            self.changes.append((start, start + len(data), changed))


def compact_html(text, hashes=None):
    """Leave attributes, links, scripts, styles and image data intact."""
    parser = _VisibleText(text, hashes)
    parser.feed(text)
    parser.close()
    for start, end, value in reversed(parser.changes):
        text = text[:start] + value + text[end:]
    return text


def compact_markdown(text, hashes=None):
    # Link destinations remain usable, including links to hash-named artifacts.
    parts = re.split(r'(\]\([^)]*\)|https?://[^\s<>]+)', text)
    return ''.join(part if i % 2 else compact_text(part, hashes)
                   for i, part in enumerate(parts))


def compact_value(value, mime='text/plain', hashes=None):
    function = compact_html if mime == 'text/html' else compact_markdown if mime == 'text/markdown' else compact_text
    if isinstance(value, list):
        # nbformat permits a multiline string to be stored as an array.
        joined = ''.join(value)
        result = function(joined, hashes)
        return result.splitlines(keepends=True) if result != joined else value
    return function(value, hashes) if isinstance(value, str) else value


def compact_mime(data, hashes=None):
    result = dict(data)
    for mime in _TEXT_MIMES:
        if mime in result:
            result[mime] = compact_value(result[mime], mime, hashes)
    return result


class _DisplayStream:
    _visualtorque_stream = True

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        self._stream.write(compact_text(text))
        return len(text)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _refresh_streams(*args, **kwargs):
    for name in ('stdout', 'stderr'):
        stream = getattr(sys, name)
        if stream is not None and not getattr(stream, '_visualtorque_stream', False):
            setattr(sys, name, _DisplayStream(stream))


def _filter_display(owner, method_name, argument):
    original = getattr(owner, method_name, None)
    if original is None or getattr(original, '_visualtorque_display', False):
        return

    @functools.wraps(original)
    def publish(*args, **kwargs):
        if args:
            args = (compact_mime(args[0]), *args[1:])
        elif argument in kwargs:
            kwargs = dict(kwargs)
            kwargs[argument] = compact_mime(kwargs[argument])
        return original(*args, **kwargs)

    publish._visualtorque_display = True
    setattr(owner, method_name, publish)


def install_display(shell=None):
    """Filter notebook display only; values, files and child-process logs stay full."""
    if shell is None:
        try:
            from IPython import get_ipython
        except ImportError:
            return False
        shell = get_ipython()
    if shell is None:
        return False
    _refresh_streams()
    _filter_display(shell.display_pub, 'publish', 'data')
    _filter_display(shell.displayhook, 'write_format_data', 'format_dict')
    if not getattr(shell, '_visualtorque_stream_hook', False):
        shell.events.register('pre_run_cell', _refresh_streams)
        shell._visualtorque_stream_hook = True
    return True
