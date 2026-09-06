# Contributing

Keep the public repository reproducible and source-focused. Do not commit run
directories, SIFs, benchmark payloads, caches, temporary files, or the optional
multi-gigabyte fitting corpus.

Before committing:

```bash
git lfs pull
make check
git status --short
```

Changes to either Dockerfile should preserve Linux AMD64 builds and must leave
the image's build-time self-test enabled. Changes to the user pipeline should
add or update dependency-free unit tests. GPU integration tests are opt-in and
are documented alongside the relevant test module.

Use a new output directory or screen name for scientific setting changes;
never edit generated manifests to force resume compatibility.
