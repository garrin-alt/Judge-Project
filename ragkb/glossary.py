"""Query-time glossary support.

A glossary maps user vocabulary onto knowledge-base vocabulary. Each entry
has a canonical term, aliases (e.g. "purple" for the Chaos domain), a
definition, and an optional source reference. At query time we scan the
prompt for terms/aliases, annotate the prompt with the canonical term so
the embedding points at the right concept, and collect the matched
definitions for the generation context.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Expansion:
    prompt: str
    matches: list[dict] = field(default_factory=list)

    @property
    def definitions(self) -> list[str]:
        return [f"{m['term']}: {m['definition']}" for m in self.matches]


def _alias_pattern(alias: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)


def expand_prompt(prompt: str, entries: list[dict], max_matches: int = 6) -> Expansion:
    """Annotate aliases with their canonical term and collect definitions.

    "purple champion" -> "purple (Chaos) champion", with the Chaos domain
    definition attached. Terms already present verbatim are not rewritten,
    but their definitions are still collected.
    """
    expanded = prompt
    matches = []
    seen_terms = set()

    for entry in entries:
        if len(matches) >= max_matches:
            break
        term = entry["term"]
        if term.lower() in seen_terms:
            continue

        matched_via = None
        for alias in entry.get("aliases", []):
            pat = _alias_pattern(alias)
            if pat.search(expanded):
                # annotate only if the canonical term isn't already there
                if not _alias_pattern(term).search(expanded):
                    expanded = pat.sub(lambda m: f"{m.group(0)} ({term})", expanded, count=1)
                matched_via = alias
                break
        if matched_via is None and _alias_pattern(term).search(prompt):
            matched_via = term

        if matched_via is not None:
            seen_terms.add(term.lower())
            matches.append({**entry, "matched": matched_via})

    return Expansion(prompt=expanded, matches=matches)
