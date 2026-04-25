"""Character discovery and per-chunk extraction.

Three-tier alias resolution:

    Tier 1a  Orthographic   NER + fuzzy clustering              (this file)
    Tier 1b  Co-reference   fastcoref title-form resolution     (this file)
    Tier 2   Semantic       LLM-assisted cluster merging        (character_merge.py)
    Tier 3   Manual         JSON overrides applied last         (data/processed/character_overrides.json)

Tier 1a finds clusters like "Harry Potter" / "Harry James Potter" via
multi-word subsetting, and absorbs OCR errors on single-word names via
fuzzy matching. Conservative — defers single-to-multi merges to later tiers.

Tier 1b runs co-reference resolution per chapter to find title-forms
("Mr. Weasley", "the boy") that resolve to named PERSON entities in
context. Aggregates across the corpus; promotes dominant resolutions to
aliases of the right canonical cluster.

Tier 2 (separate file) handles semantic aliases the orthographic and coref
passes can't catch — "Voldemort" / "He Who Must Not Be Named" / "Tom Riddle".

See ADR-0006.
"""

import re
from collections import Counter, defaultdict
from typing import Iterable

from rapidfuzz import fuzz
from spacy.language import Language

from horcrux.models import Chapter

# Pronouns and trivial words to skip when looking at coref-cluster mentions.
# These don't make useful aliases for chunk-tagging.
_PRONOUNS = frozenset({
    "he", "she", "him", "her", "his", "hers", "himself", "herself",
    "they", "them", "their", "theirs", "themselves",
    "it", "its", "itself", "this", "that", "these", "those",
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "we", "us", "our", "ours", "ourselves",
    "who", "whom", "whose", "which", "what",
})

# Generic title words that don't constitute a "shared identity" between
# alias and target. "Mr. Filch" sharing "filch" with "Filch" is meaningful;
# "Mr. Dursley" sharing "mr." with "Mr. Mason" is not.
_GENERIC_TITLES = frozenset({
    "mr.", "mr", "mrs.", "mrs", "miss", "ms.", "ms", "master",
    "lord", "lady", "sir", "madam", "madame",
    "aunt", "uncle", "auntie",
    "professor", "prof.", "prof", "dr.", "dr", "doctor",
    "the", "a", "an",
    "father", "mother", "dad", "mom", "mum", "daddy", "mommy", "mummy",
    "boy", "girl", "man", "woman", "kid", "child",
})

# Conjunctions in alias candidates signal multi-person mentions ("Harry
# and Ron"), which can't legitimately alias either character alone.
_CONJUNCTIONS = frozenset({"and", "or", "&", "nor"})


def _significant_tokens(name: str) -> set[str]:
    """Lowercased tokens with generic titles + articles stripped.

    `Mr. Filch`         → {'filch'}
    `Aunt Petunia`      → {'petunia'}
    `the boy`           → {} (all generic)
    `Albus Dumbledore`  → {'albus', 'dumbledore'}
    """
    return {t for t in name.lower().split() if t not in _GENERIC_TITLES}


def _shares_significant_token(alias: str, target: str) -> bool:
    """Conservative gate for Tier 1b coref resolutions.

    Allows "Mr. Filch" → Filch, blocks "He-Who-Must-Not-Be-Named" →
    Dumbledore. Semantic aliases (no shared tokens) are deliberately
    deferred to Tier 2 LLM, which has the context to handle them.
    """
    a = _significant_tokens(alias)
    t = _significant_tokens(target)
    if not a or not t:
        return False
    return bool(a & t)


# ── NER pass ──────────────────────────────────────────────────────

def extract_person_mentions(text: str, nlp: Language) -> list[str]:
    """Return all PERSON entity mentions in `text`, in document order.

    Whitespace-stripped. Duplicates are kept (caller aggregates).
    """
    doc = nlp(text)
    return [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]


_POSSESSIVE_SUFFIX = re.compile(r"['’]s$", re.IGNORECASE)
_INTERNAL_WHITESPACE = re.compile(r"\s+")


def _normalise_mention(name: str) -> str:
    """Normalise an extracted mention: collapse internal whitespace, strip
    possessive suffixes, trim outer whitespace.

    NER often captures names that span line breaks ("Professor\\nMcGonagall")
    when a name wraps in the source text. Collapsing whitespace makes those
    cluster correctly with the same name on a single line.

    'Harry's'              → 'Harry'
    'Professor\\nMcGonagall' → 'Professor McGonagall'
    """
    s = _POSSESSIVE_SUFFIX.sub("", name.strip())
    s = _INTERNAL_WHITESPACE.sub(" ", s)
    return s


def count_mentions(chapters: Iterable[Chapter], nlp: Language) -> Counter[str]:
    """Aggregate PERSON mention counts across all chapters.

    Possessive suffixes ('s, 's) are stripped before counting so possessive
    forms fold into their root name's count.

    Uses `nlp.pipe` for batched throughput — much faster than calling
    `nlp(text)` in a Python loop.
    """
    counter: Counter[str] = Counter()
    texts = [chapter.text for chapter in chapters]
    for doc in nlp.pipe(texts, batch_size=4):
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = _normalise_mention(ent.text)
                if name:
                    counter[name] += 1
    return counter


# ── Fuzzy clustering ──────────────────────────────────────────────

def _is_alias_pair(a: str, b: str, threshold: int) -> bool:
    """Decide if two mention strings should cluster — conservative.

    Trigger conditions:
      1. **Multi-word ↔ multi-word subset**.  `Harry Potter` ⊂ `Harry
         James Potter` clusters; `Mr. Weasley` ↔ `Mrs. Weasley` does not.
      2. **Single-word ↔ single-word fuzz**.  `Hermione` ≈ `Hcrmione`
         (OCR error) clusters.

    Deliberately does NOT cluster single-word ↔ multi-word pairs — that
    avoids the family-name failure mode (`Weasley` pulling in every
    Ron/Fred/George/Percy/Ginny Weasley as one cluster). Single-to-multi
    merges are deferred to the Tier 2 LLM step where the model has the
    semantic context to distinguish 'Harry → Harry Potter' (same person)
    from 'Weasley → Ron Weasley' (just a family name).
    """
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return False

    # Multi-word ↔ multi-word: subset matching is safe.
    if len(a_words) > 1 and len(b_words) > 1:
        return a_words.issubset(b_words) or b_words.issubset(a_words)

    # Single-word ↔ single-word: char-level fuzz catches OCR errors.
    if len(a_words) == 1 and len(b_words) == 1:
        return fuzz.ratio(a.lower(), b.lower()) >= threshold

    # Single-word ↔ multi-word: defer to Tier 2.
    return False


def cluster_aliases(
    counts: Counter[str],
    *,
    min_count: int = 3,
    similarity_threshold: int = 85,
) -> dict[str, list[str]]:
    """Cluster mention strings into canonical → aliases groups.

    Args:
        counts: mention frequency from `count_mentions`.
        min_count: drop mentions appearing fewer than this many times
            (cuts the long tail of NER noise).
        similarity_threshold: 0-100; rapidfuzz ratio threshold for
            clustering by character-level similarity.

    Returns:
        dict[canonical, list[aliases sorted by frequency desc]]
        Canonical = most-frequent mention in each cluster.

    Algorithm: union-find over pairwise similarity.
    """
    candidates = [name for name, c in counts.items() if c >= min_count]
    if not candidates:
        return {}

    # Union-find with path compression.
    parent: dict[str, str] = {name: name for name in candidates}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # O(n²) pairwise check. For ~1000 candidates that's 500k comparisons —
    # rapidfuzz handles this in well under a second.
    for i, a in enumerate(candidates):
        for b in candidates[i + 1 :]:
            if _is_alias_pair(a, b, similarity_threshold):
                union(a, b)

    # Group by root.
    groups: dict[str, list[str]] = {}
    for name in candidates:
        groups.setdefault(find(name), []).append(name)

    # Pick canonical = most frequent in each group.
    result: dict[str, list[str]] = {}
    for members in groups.values():
        canonical = max(members, key=lambda n: counts[n])
        result[canonical] = sorted(members, key=lambda n: -counts[n])

    return result


# ── Per-chunk extraction ──────────────────────────────────────────

# ── Tier 1b: co-reference resolution ─────────────────────────────

def _is_meaningful_mention(text: str) -> bool:
    """Filter coref-cluster mentions down to ones useful as aliases.

    Drops:
      - pronouns ("he", "his", "the")
      - one- or two-character noise
      - leading articles ("a man", "the boy") — keep "boy who lived" not "the boy"
    """
    text = text.strip().lower()
    if len(text) < 3:
        return False
    if text in _PRONOUNS:
        return False
    # Articles + single common nouns aren't useful labels either.
    first_word = text.split()[0] if text.split() else ""
    if first_word in {"a", "an", "the"} and len(text.split()) <= 2:
        return False
    return True


def _is_alias_candidate(text: str, *, max_words: int = 4) -> bool:
    """Stricter than `_is_meaningful_mention`: filter to title-like forms.

    Coref clusters legitimately contain prose fragments ("the round-faced
    boy", "Harry and Ron, who…") that are valid co-references but useless
    as alias keys for chunk tagging. Restrict to compact, title-like
    phrases referring to a single person.

    Rules on top of `_is_meaningful_mention`:
      - ≤ max_words tokens long
      - no commas, semicolons, or quote marks (running prose)
      - no conjunctions (multi-person mentions)
    """
    if not _is_meaningful_mention(text):
        return False
    if any(ch in text for ch in ',;"\'“”‘’'):
        return False
    if len(text.split()) > max_words:
        return False
    if any(w in _CONJUNCTIONS for w in text.lower().split()):
        return False
    return True


def resolve_coref_aliases(
    chapters: Iterable[Chapter],
    nlp: Language,
    coref_model,
    *,
    dominance_ratio: float = 2.0,
    min_corpus_occurrences: int = 2,
    max_alias_words: int = 4,
) -> dict[str, list[str]]:
    """Use co-reference resolution to find which named PERSON each
    title-form / partial-reference resolves to across the corpus.

    Conservative — drops noisy cases rather than guessing:
      - Coref clusters containing >1 distinct named PERSON are skipped
        entirely (cross-character coref errors corrupt attribution).
      - Mentions matching NER PERSON entities by *text* (case-insensitive)
        are treated as named entities even when char spans differ slightly.
      - Alias candidates are length-capped and reject prose punctuation.
      - Single-instance attributions across the whole corpus are dropped.

    Args:
        chapters: source chapters with cleansed text.
        nlp: spaCy Language pipeline (for PERSON entity spans).
        coref_model: a fastcoref `FCoref` instance.
        dominance_ratio: a resolution is "dominant" when the top named
            person appears ≥ ratio× more often than the second-most.
        min_corpus_occurrences: drop title-forms that resolved fewer than
            this many times across the corpus (likely noise).
        max_alias_words: aliases longer than this are rejected (prose, not
            stable identifiers).

    Returns:
        dict[title_form, list[canonical names]]
            len == 1 → confident dominant resolution
            len > 1  → ambiguous, multi-tag at chunk time
    """
    resolutions: dict[str, Counter[str]] = defaultdict(Counter)

    chapters_list = list(chapters)
    if not chapters_list:
        return {}

    texts = [c.text for c in chapters_list]
    preds = coref_model.predict(texts=texts)

    for chapter, pred in zip(chapters_list, preds, strict=True):
        text = chapter.text

        # NER PERSON spans AND a lower-cased text set for boundary-tolerant
        # matching (coref and NER often disagree on exact char boundaries).
        doc = nlp(text)
        person_spans: dict[tuple[int, int], str] = {}
        person_text_set: set[str] = set()
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            normalised = _normalise_mention(ent.text)
            if not normalised:
                continue
            person_spans[(ent.start_char, ent.end_char)] = normalised
            person_text_set.add(normalised.lower())

        for cluster in pred.get_clusters(as_strings=False):
            # Use a SET for named — multi-PERSON clusters get rejected later.
            named_in_cluster: set[str] = set()
            alias_candidates: list[str] = []

            for span_start, span_end in cluster:
                raw = text[span_start:span_end].strip()
                normalised = _normalise_mention(raw)
                if not normalised:
                    continue

                # Span-exact match OR text match → known PERSON entity.
                is_person = (
                    (span_start, span_end) in person_spans
                    or normalised.lower() in person_text_set
                )
                if is_person:
                    named_in_cluster.add(normalised)
                elif _is_alias_candidate(normalised, max_words=max_alias_words):
                    alias_candidates.append(normalised)

            # Reject coref clusters with multiple distinct named persons —
            # coref made an error and we can't recover the right attribution.
            if len(named_in_cluster) != 1:
                continue

            owner = next(iter(named_in_cluster))
            for mention in alias_candidates:
                # Defence in depth: if the candidate text is still a known
                # PERSON elsewhere, don't attribute it as an alias.
                if mention.lower() in person_text_set:
                    continue
                # Tier 1b is strictly orthographic — alias must share at
                # least one significant token with the target. Semantic
                # aliases ("He Who Must Not Be Named" → Voldemort) are
                # deferred to Tier 2 LLM where they're handled correctly.
                if not _shares_significant_token(mention, owner):
                    continue
                resolutions[mention][owner] += 1

    # Compress to dominant-or-ambiguous; drop low-occurrence noise.
    result: dict[str, list[str]] = {}
    for title, counts in resolutions.items():
        ranked = counts.most_common()
        if not ranked:
            continue
        top_name, top_count = ranked[0]
        if top_count < min_corpus_occurrences:
            continue
        if len(ranked) == 1 or top_count >= dominance_ratio * ranked[1][1]:
            result[title] = [top_name]
        else:
            result[title] = [name for name, _ in ranked]

    return result


def claim_single_word_clusters(
    clusters: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Merge single-word clusters into multi-word clusters that *uniquely*
    own them as a significant token.

    Rule:
      A single-word cluster `S` merges into a multi-word cluster `M` iff
      both of:
        - `S`'s canonical contains exactly one significant (non-generic)
          token `t`
        - exactly one multi-word cluster's canonical contains `t` as one
          of its significant tokens

    The "exactly one" gate is what distinguishes a personal name (only
    one multi-word cluster owns it — `Hermione` → `Hermione Granger`)
    from a family/group name (multiple multi-word clusters share it —
    `Weasley` is part of `Ron Weasley`, `Fred Weasley`, ... ; ambiguous,
    don't merge).

    Canonical preference: when merging, the multi-word form wins. Yields
    human-readable cluster names in reports (`Harry Potter`, not `Harry`)
    even though the single-word has higher frequency. Aliases from the
    single-word cluster are appended.

    Pure function; doesn't mutate input. Order of execution within a
    single discovery run doesn't matter — uses the original cluster
    membership snapshot, not a live-mutating one.
    """
    # Index: significant_token → list of multi-word CANONICALS containing it.
    multiword_owners: dict[str, list[str]] = defaultdict(list)
    for canonical in clusters:
        if len(canonical.split()) > 1:
            for token in _significant_tokens(canonical):
                multiword_owners[token].append(canonical)

    new_clusters = {k: list(v) for k, v in clusters.items()}

    for canonical in list(clusters.keys()):
        if len(canonical.split()) != 1:
            continue
        sig_tokens = _significant_tokens(canonical)
        if len(sig_tokens) != 1:
            continue
        token = next(iter(sig_tokens))
        owners = multiword_owners.get(token, [])
        if len(owners) != 1:
            continue

        target = owners[0]
        if target not in new_clusters or canonical not in new_clusters:
            continue

        for alias in new_clusters[canonical]:
            if alias not in new_clusters[target]:
                new_clusters[target].append(alias)
        del new_clusters[canonical]

    return new_clusters


def merge_coref_into_clusters(
    clusters: dict[str, list[str]],
    coref_resolutions: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Fold coref-resolved aliases into the existing cluster dict.

    For each (title_form, [resolved_names]):
      - len([resolved_names]) == 1 (dominant): add title_form as alias to
        the cluster containing that name; remove title_form's own cluster
        if it exists.
      - len > 1 (ambiguous): leave alone — the multi-tag case is handled
        at extraction time, not at cluster build time.

    Returns a new dict; doesn't mutate input.
    """
    new_clusters = {k: list(v) for k, v in clusters.items()}

    for title, resolved in coref_resolutions.items():
        if len(resolved) != 1:
            continue
        target_name = resolved[0]

        # Find which cluster owns target_name (canonical or any alias).
        target_canonical = None
        for canonical, aliases in new_clusters.items():
            if target_name == canonical or target_name in aliases:
                target_canonical = canonical
                break
        if target_canonical is None:
            continue

        if title not in new_clusters[target_canonical]:
            new_clusters[target_canonical].append(title)

        # If title was its own cluster, dissolve it.
        if title in new_clusters and title != target_canonical:
            del new_clusters[title]

    return new_clusters


# ── Per-chunk extraction ──────────────────────────────────────────

def extract_characters(
    text: str,
    alias_dict: dict[str, list[str]],
) -> list[str]:
    """Find canonical character names mentioned in `text`.

    Whole-word substring matching. Each canonical is listed at most
    once even if multiple of its aliases appear. Output is sorted for
    deterministic comparison.
    """
    if not text or not alias_dict:
        return []

    text_lower = text.lower()
    found: set[str] = set()

    for canonical, aliases in alias_dict.items():
        for alias in aliases:
            pattern = r"\b" + re.escape(alias.lower()) + r"\b"
            if re.search(pattern, text_lower):
                found.add(canonical)
                break  # one matched alias per canonical is enough

    return sorted(found)
