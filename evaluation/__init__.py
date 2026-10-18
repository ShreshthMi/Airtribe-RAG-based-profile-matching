"""Retrieval-quality and latency measurement for the resume matcher.

`metrics` holds pure ranking metrics; `harness` drives the matcher over the
hand-labelled job descriptions and reports accuracy and timings.

The package is named `evaluation` rather than `eval` so it cannot shadow the
builtin of that name.
"""
