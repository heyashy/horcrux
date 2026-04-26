"""Corpus preparation — PDF → cleansed pages → chapters → chunks.

This subpackage owns everything between the raw PDF and the chunks
that go into Qdrant. Roughly:

    pdf → ocr → cleansing → chapters → chunking → embedding

Each module is one stage of that pipeline. Character discovery
(`characters`) is a side-pipeline that runs alongside chunking,
producing the alias dictionary chunks reference for character payload.

Public surface — use module-qualified imports for clarity:

    from horcrux.corpus.chunking import chunk_chapter
    from horcrux.corpus.embedding import encode_query, encode_passages
    from horcrux.corpus.characters import extract_characters
"""
