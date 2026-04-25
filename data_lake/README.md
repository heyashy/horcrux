# data_lake/

Raw, unstructured source data for the lab. Treated as the system of record —
everything in `data/`, `qdrant_storage/`, and the embeddings index is
*derivable* from what lives here.

## What goes here

A single PDF: `corpus.pdf`. Drop your legally-obtained copy of the corpus you
want to research into this folder. The lab is built and tested against a
~3600-page literary corpus, but any text-bearing PDF will work — the OCR
pipeline tolerates image-based PDFs.

## What's git-tracked

Just this README. Everything else in this directory is `.gitignore`d for
copyright and bandwidth reasons. If you fork the repo, this folder will be
empty until you populate it.

## Why "data lake"

It's the canonical term for raw, unstructured, source-of-truth data. The lab
deliberately separates the lake (this folder — immutable inputs) from
*derived* layers (OCR'd text, chunks, embeddings — all regeneratable). If the
ingest pipeline corrupts itself, you delete `data/`, `qdrant_storage/`, and
`horcrux.db` and re-run; the lake is untouched.
