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
from collections.abc import Iterable

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

# Stop-modifiers that NER captures as part of a "PERSON" mention but which
# aren't part of the character's identity. Examples from real corpus output:
#   "Voldemort himself"  → noise, the name is "Voldemort"
#   "Dear Harry"         → noise, the name is "Harry"
#   "only Crookshanks"   → noise, the name is "Crookshanks"
#   "poor old Ripper"    → noise, the name is "Ripper"
#   "Harry blankly"      → noise (NER mistakenly captured a verb adverb)
#
# Word-boundary substring matching at chunk-extraction time still finds
# the underlying name inside these phrases, so we don't need them as
# explicit aliases — we just need to reject them as alias candidates.
_STOP_MODIFIERS = frozenset({
    # Reflexive pronouns (third-person)
    "himself", "herself", "itself", "themselves", "oneself",
    # Affection / address
    "dear", "dearest", "darling", "sweet",
    # Restrictive / focus
    "only", "even", "just", "alone", "also", "merely",
    # Quality adjectives that drift onto names in fiction
    "poor", "old", "young", "dear",
    # Adverbs that NER occasionally captures as part of "Name + adverb"
    # tokenisations (e.g. "Harry blankly", "Ron grumpily"). The full set
    # is open-ended; these are the ones that survived discovery.
    "blankly", "grumpily", "muttered", "wearily", "happily", "hastily",
    "calmly", "cautiously", "heavily", "weakly", "thickly", "spat",
    "hoarsely",
})


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
    Dumbledore. Semantic aliases (no shared tokens) are deferred to
    manual override via Tier 3.

    Note: this rule alone allows `Albus Dumbledore` → `Ariana Dumbledore`
    (both share 'dumbledore', looks like a valid attribution by tokens
    alone). The corpus-wide person_text_set check in `resolve_coref_aliases`
    is the second line of defence — blocking *any* NER-tagged PERSON
    entity from being attributed as an alias of another. See Finding 10.

    A purely orthographic rule cannot distinguish:
      - Albus / Ariana Dumbledore (block — different people)
      - Mad-Eye / Alastor Moody  (allow — same person, nickname)
    Both have "shared surname + dissimilar unique part." The right tool
    for that decision is world knowledge, not orthography. Tier 3 manual
    override handles whichever cases the corpus-wide person check gets
    wrong.
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

    Multi-token NER mentions are filtered through `_is_alias_candidate`
    BEFORE counting — drops prose-fragment false positives like
    "Harry blankly", "Harry, Ron", "Voldemort himself" that would
    otherwise survive as standalone clusters and pollute Tier 1c's
    owner index. Single-token names (`Harry`, `Hogwarts`) are kept; the
    stop-modifier filter only matches multi-token forms anyway.

    Uses `nlp.pipe` for batched throughput.
    """
    counter: Counter[str] = Counter()
    texts = [chapter.text for chapter in chapters]
    for doc in nlp.pipe(texts, batch_size=4):
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            name = _normalise_mention(ent.text)
            if not name:
                continue
            # Reject multi-token prose-fragment false positives. Single-
            # token names always pass (they have no stop-modifier or
            # conjunction tokens by definition).
            if len(name.split()) > 1 and not _is_alias_candidate(name, max_words=4):
                continue
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
    min_anchor_count: int = 5,
) -> dict[str, list[str]]:
    """Cluster mention strings into canonical → aliases groups.

    Args:
        counts: mention frequency from `count_mentions`.
        min_count: drop mentions appearing fewer than this many times
            (cuts the long tail of NER noise).
        similarity_threshold: 0-100; rapidfuzz ratio threshold for
            clustering by character-level similarity.
        min_anchor_count: a multi-word name with 3+ significant tokens
            must have at least this many mentions to act as a *bridge*
            in subset clustering. Prevents transitive merges via rare
            full-name forms — e.g. "Harry James Potter" with count=2
            won't bridge "Harry Potter" and "James Potter" into one
            cluster (they're different people; the bridge is just a rare
            full-name form). Doesn't apply to 2-significant-token names
            since those don't bridge two distinct shorter forms.

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
            if not _is_alias_pair(a, b, similarity_threshold):
                continue

            # Anchor-count check: when a multi-word subset merge is
            # mediated by a 3+-significant-token "anchor", require that
            # anchor to be common enough to be canonical. Stops rare
            # full-name forms (like "Harry James Potter" with count=2)
            # from bridging two otherwise-distinct two-word names
            # ("Harry Potter" and "James Potter"). See Finding 9.
            a_sig = _significant_tokens(a)
            b_sig = _significant_tokens(b)
            if len(a_sig) > 1 and len(b_sig) > 1:
                # Identify the larger (potential anchor)
                larger_name = a if len(a_sig) > len(b_sig) else b
                larger_sig = a_sig if larger_name is a else b_sig
                if len(larger_sig) >= 3 and counts[larger_name] < min_anchor_count:
                    continue

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
    boy", "Harry and Ron, who…") and modifier+name pairs ("Voldemort
    himself", "Dear Harry") that are valid co-references but useless as
    alias keys for chunk tagging. Restrict to compact, title-like phrases
    referring to a single person.

    Rules on top of `_is_meaningful_mention`:
      - ≤ max_words tokens long
      - no commas, semicolons, or quote marks (running prose)
      - no conjunctions (multi-person mentions)
      - no stop-modifiers (modifier+name forms that aren't real aliases)
    """
    if not _is_meaningful_mention(text):
        return False
    if any(ch in text for ch in ',;"\'“”‘’'):
        return False
    if len(text.split()) > max_words:
        return False
    words = text.lower().split()
    if any(w in _CONJUNCTIONS for w in words):
        return False
    if any(w in _STOP_MODIFIERS for w in words):
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
    import gc
    import logging as _logging

    _log = _logging.getLogger(__name__)

    resolutions: dict[str, Counter[str]] = defaultdict(Counter)

    chapters_list = list(chapters)
    if not chapters_list:
        return {}

    total = len(chapters_list)

    # PRE-PASS: NER over every chapter, cache per-chapter spans, and
    # accumulate a CORPUS-WIDE person_text_set. Any NER PERSON entity
    # that appears anywhere in the corpus is a known character, and must
    # not be attributed as an alias of another character — even in
    # chapters where NER didn't tag it locally. See Finding 10: per-
    # chapter person sets leak attributions across same-surname entities
    # (Albus Dumbledore got merged into the Ariana Dumbledore cluster
    # because chapters about Ariana didn't tag Albus locally).
    _log.info("NER pre-pass over %d chapters", total)
    chapter_person_spans: list[dict[tuple[int, int], str]] = []
    corpus_person_texts: set[str] = set()
    for chapter in chapters_list:
        doc = nlp(chapter.text)
        spans: dict[tuple[int, int], str] = {}
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            normalised = _normalise_mention(ent.text)
            if not normalised:
                continue
            spans[(ent.start_char, ent.end_char)] = normalised
            corpus_person_texts.add(normalised.lower())
        chapter_person_spans.append(spans)

    # Stream chapters one at a time. Passing all texts in a single
    # `predict(texts=[...])` call OOMs on full-corpus runs (each chapter
    # becomes a model tensor; ~200 chapters × ~30KB text simultaneously
    # exceeds GPU+CPU memory). Per-chapter prediction keeps peak memory
    # bounded to one chapter's model state at a time.
    for i, chapter in enumerate(chapters_list):
        if i % 10 == 0:
            _log.info("coref pass: chapter %d/%d", i + 1, total)

        preds = coref_model.predict(texts=[chapter.text])
        pred = preds[0]
        text = chapter.text

        # Per-chapter NER cached from the pre-pass.
        person_spans = chapter_person_spans[i]

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
                    or normalised.lower() in corpus_person_texts
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
                if mention.lower() in corpus_person_texts:
                    continue
                # Tier 1b is strictly orthographic — alias must share at
                # least one significant token with the target. Semantic
                # aliases ("He Who Must Not Be Named" → Voldemort) are
                # deferred to Tier 2 LLM where they're handled correctly.
                if not _shares_significant_token(mention, owner):
                    continue
                resolutions[mention][owner] += 1

        # Per-chapter cleanup. Without this, model-state tensors
        # accumulate in memory across iterations and OOM on full-corpus
        # runs. (NER docs were processed in the pre-pass and only spans
        # are retained per-chapter — much smaller than full Doc objects.)
        del preds, pred
        if (i + 1) % 5 == 0:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

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


# ── Identifier model: separate ID from surface form ─────────────
# Industry-standard pattern: opaque identifier decouples identity from
# display label, supporting renames, multi-language, and disambiguation
# without rewriting downstream consumers. See Finding 15.

_SLUG_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_SLUG_WHITESPACE = re.compile(r"\s+")


def slugify(name: str) -> str:
    """Generate a stable, opaque identifier from a display name.

    Deterministic — same input always produces same slug. Re-running
    discovery doesn't shuffle character IDs.

    `Harry Potter`        → `harry_potter`
    `Albus Dumbledore`    → `albus_dumbledore`
    `T. M. Riddle`        → `t_m_riddle`
    `Mary GrandPré`       → `mary_grandpré`     (Unicode preserved)
    """
    slug = _SLUG_PUNCT.sub("", name.lower()).strip()
    slug = _SLUG_WHITESPACE.sub("_", slug)
    return slug


def to_id_indexed(clusters: dict[str, list[str]]) -> dict[str, dict]:
    """Convert canonical-keyed cluster dict to ID-keyed entity records.

    Output shape:
        {
            "<slug>": {"label": "<canonical>", "aliases": [...]},
            ...
        }

    Slug collisions (different canonicals producing the same slug)
    are disambiguated by appending a numeric suffix, deterministically.
    """
    result: dict[str, dict] = {}
    for canonical, aliases in clusters.items():
        slug = slugify(canonical)
        if not slug:
            continue
        # Disambiguate collisions by suffix.
        if slug in result:
            i = 2
            while f"{slug}_{i}" in result:
                i += 1
            slug = f"{slug}_{i}"
        result[slug] = {
            "label": canonical,
            "aliases": list(aliases),
        }
    return result


def lookup_label(alias_dict: dict[str, dict], char_id: str) -> str | None:
    """Return the display label for a character ID, or None if unknown."""
    record = alias_dict.get(char_id)
    return record["label"] if record else None


def apply_overrides(
    alias_dict: dict[str, dict],
    overrides: dict,
) -> dict[str, dict]:
    """Apply Tier 3 manual overrides to an ID-indexed alias dict.

    Operations supported:

    - ``drop`` — list of character IDs to remove entirely. For NER false
      positives (places, objects, magical concepts, prose fragments).

    - ``force_merge`` — list of merge groups, each ``[primary_id, *secondary_ids]``.
      Each secondary's label and aliases are added to the primary; secondaries
      are then removed. For semantic aliases that no orthographic algorithm
      could derive (Padfoot → Sirius, Tom Riddle → Voldemort).

    Pure function; doesn't mutate input.
    """
    # Deep-copy so we don't mutate caller's dict.
    result: dict[str, dict] = {
        cid: {"label": rec["label"], "aliases": list(rec["aliases"])}
        for cid, rec in alias_dict.items()
    }

    for char_id in overrides.get("drop", []):
        result.pop(char_id, None)

    for group in overrides.get("force_merge", []):
        if len(group) < 2:
            continue
        primary_id = group[0]
        if primary_id not in result:
            continue
        primary_aliases = result[primary_id]["aliases"]
        for secondary_id in group[1:]:
            secondary = result.pop(secondary_id, None)
            if secondary is None:
                continue
            # Merge label + aliases of secondary into primary's alias list.
            for alias in [secondary["label"], *secondary["aliases"]]:
                if alias not in primary_aliases:
                    primary_aliases.append(alias)

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
    # Only ≤ 2-significant-token canonicals are indexed as "owners".
    # Three-or-more-significant-token forms are full-name expansions
    # ("Harry James Potter", "Albus Wulfric Brian Dumbledore"), not
    # personal-name anchors. Including them as owners blocks shorter
    # forms from folding (e.g. "Harry" sees two owners — "Harry Potter"
    # and "Harry James Potter" — and refuses to fold even though
    # "Harry Potter" is the obvious anchor). Same principle as Finding 9
    # at Tier 1a; applied here at Tier 1c.
    multiword_owners: dict[str, list[str]] = defaultdict(list)
    for canonical in clusters:
        if len(canonical.split()) > 1:
            sig_tokens = _significant_tokens(canonical)
            if len(sig_tokens) > 2:
                continue
            for token in sig_tokens:
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
    alias_dict: dict[str, dict],
) -> list[str]:
    """Find character IDs mentioned in `text`.

    Returns sorted list of character IDs (slugs) — NOT display labels.
    Caller does ID→label lookup for display via `lookup_label`. This
    decouples identity from surface form (industry standard for entity
    dictionaries — see Finding 15).

    Whole-word substring matching, case-insensitive. Each character is
    listed at most once even if multiple of its aliases appear.
    """
    if not text or not alias_dict:
        return []

    text_lower = text.lower()
    found: set[str] = set()

    for char_id, record in alias_dict.items():
        for alias in record.get("aliases", []):
            pattern = r"\b" + re.escape(alias.lower()) + r"\b"
            if re.search(pattern, text_lower):
                found.add(char_id)
                break  # one matched alias per character is enough

    return sorted(found)
