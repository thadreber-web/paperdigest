from importlib.metadata import PackageNotFoundError, version

# Distribution name on PyPI is `arxivdigest` (the name `paperdigest` is reserved
# by PyPI's name-similarity guard for an unrelated project); the import package
# stays `paperdigest`. Look up real installed metadata by current name first,
# then the legacy name so older editable installs still report their version.
for _dist_name in ("arxivdigest", "paperdigest"):
    try:
        __version__ = version(_dist_name)
        break
    except PackageNotFoundError:
        continue
else:  # running from a source tree without installation
    __version__ = "0.0.0"
