"""Run bounded local NLP feasibility checks; this is not CDA validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
FIXTURES = json.loads((ROOT / "fixtures.json").read_text(encoding="utf-8"))

SUBJECT_DEPS = {"nsubj", "nsubj:pass", "nsubjpass", "csubj", "csubj:pass"}
PASSIVE_DEPS = {"auxpass", "aux:pass", "nsubjpass", "nsubj:pass", "csubj:pass"}
PARTICIPANT_DEPS = SUBJECT_DEPS | {
    "obj", "dobj", "iobj", "pobj", "obl", "agent", "nmod", "nmod:agent",
}
ACTOR_ENTITY_LABELS = {"PERSON", "PER", "ORG", "NORP", "GPE", "LOC", "FAC", "MISC"}
MODAL_LEXEMES = {
    "en": {"must", "can", "may", "should", "will", "would", "could", "might", "shall"},
    "es": {"deber", "poder"},
    "ar": {"وجب", "يجب", "يمكن", "أمكن", "استطاع", "ينبغي"},
}


def _status(ok: bool) -> str:
    return "SUPPORTED" if ok else "LIMITED"


def _candidate_summary(
    per_fixture: dict[str, list[int]],
) -> tuple[dict[str, bool], dict[str, Any]]:
    counts = {name: sum(values) for name, values in per_fixture.items()}
    hits = {name: sum(value > 0 for value in values) for name, values in per_fixture.items()}
    capabilities = {name: count > 0 for name, count in counts.items()}
    return capabilities, {"candidate_counts": counts, "candidate_fixture_hits": hits}


def _spacy(language: str, package: str) -> dict[str, Any]:
    import spacy

    pipeline = spacy.load(package)
    docs = list(pipeline.pipe(FIXTURES[language]))
    tokens = [token for doc in docs for token in doc]
    sentences = [list(doc.sents) for doc in docs]
    all_entities = [entity for doc in docs for entity in doc.ents]
    actor_entities = [entity for entity in all_entities if entity.label_ in ACTOR_ENTITY_LABELS]
    per_fixture = {
        "agency_candidates": [],
        "passive_candidates": [],
        "modal_candidates": [],
        "social_actor_candidates": [],
    }
    for doc in docs:
        doc_tokens = list(doc)
        per_fixture["agency_candidates"].append(
            sum(token.dep_ in SUBJECT_DEPS for token in doc_tokens)
        )
        per_fixture["passive_candidates"].append(
            sum(token.dep_ in PASSIVE_DEPS for token in doc_tokens)
        )
        per_fixture["modal_candidates"].append(
            sum(
                token.lemma_.casefold() in MODAL_LEXEMES[language]
                or token.text.casefold() in MODAL_LEXEMES[language]
                for token in doc_tokens
            )
        )
        participant_heads = {
            token.i
            for token in doc_tokens
            if token.pos_ in {"NOUN", "PROPN", "PRON"}
            and token.dep_ in PARTICIPANT_DEPS
        }
        actor_entity_spans = {
            (entity.start, entity.end)
            for entity in doc.ents
            if entity.label_ in ACTOR_ENTITY_LABELS
        }
        per_fixture["social_actor_candidates"].append(
            len(participant_heads) + len(actor_entity_spans)
        )

    candidate_capabilities, candidate_evidence = _candidate_summary(per_fixture)
    capability = {
        "sentence_segmentation": bool(docs) and all(sentences),
        "tokenization": bool(tokens),
        "lemmatization": bool(tokens) and all(token.lemma_ for token in tokens),
        "pos": bool(tokens) and all(token.pos_ for token in tokens),
        "morphology": bool(tokens) and any(token.morph.to_dict() for token in tokens),
        "dependencies": bool(tokens) and all(token.dep_ for token in tokens),
        "ner": pipeline.has_pipe("ner") and bool(all_entities),
        **candidate_capabilities,
    }
    return {
        "backend": "spaCy",
        "model": package,
        "capabilities": {name: _status(value) for name, value in capability.items()},
        "assurance": "TESTED",
        "fixture_count": len(docs),
        "evidence": {
            "sentence_count": sum(len(doc_sentences) for doc_sentences in sentences),
            "token_count": len(tokens),
            "morphology_feature_token_count": sum(bool(token.morph.to_dict()) for token in tokens),
            "ner_entity_count": len(all_entities),
            "ner_actor_entity_count": len(actor_entities),
            **candidate_evidence,
        },
    }


def _stanza_arabic() -> dict[str, Any]:
    import stanza
    from stanza.pipeline.core import DownloadMethod

    pipeline = stanza.Pipeline(
        "ar",
        model_dir=str(ROOT / "models"),
        package="default",
        processors="tokenize,pos,lemma,depparse,ner",
        download_method=DownloadMethod.NONE,
        use_gpu=False,
        verbose=False,
    )
    docs = [pipeline(text) for text in FIXTURES["ar"]]
    words = [word for doc in docs for sentence in doc.sentences for word in sentence.words]
    all_entities = [
        entity
        for doc in docs
        for entity in (getattr(doc, "entities", ()) or ())
    ]
    actor_entities = [
        entity
        for entity in all_entities
        if getattr(entity, "type", None) in ACTOR_ENTITY_LABELS
    ]
    per_fixture = {
        "agency_candidates": [],
        "passive_candidates": [],
        "modal_candidates": [],
        "social_actor_candidates": [],
    }
    for doc in docs:
        doc_words = [word for sentence in doc.sentences for word in sentence.words]
        per_fixture["agency_candidates"].append(
            sum(word.deprel in SUBJECT_DEPS for word in doc_words)
        )
        per_fixture["passive_candidates"].append(
            sum(word.deprel in PASSIVE_DEPS for word in doc_words)
        )
        per_fixture["modal_candidates"].append(
            sum(
                (word.lemma or "").casefold() in MODAL_LEXEMES["ar"]
                or word.text.casefold() in MODAL_LEXEMES["ar"]
                for word in doc_words
            )
        )
        participant_heads = {
            (word.id, word.text)
            for word in doc_words
            if word.upos in {"NOUN", "PROPN", "PRON"}
            and word.deprel in PARTICIPANT_DEPS
        }
        actor_entity_spans = {
            (entity.text, getattr(entity, "type", None))
            for entity in (getattr(doc, "entities", ()) or ())
            if getattr(entity, "type", None) in ACTOR_ENTITY_LABELS
        }
        per_fixture["social_actor_candidates"].append(
            len(participant_heads) + len(actor_entity_spans)
        )

    candidate_capabilities, candidate_evidence = _candidate_summary(per_fixture)
    capability = {
        "sentence_segmentation": bool(docs) and all(doc.sentences for doc in docs),
        "tokenization": bool(words),
        "lemmatization": bool(words) and all(word.lemma for word in words),
        "pos": bool(words) and all(word.upos for word in words),
        "morphology": bool(words) and any(word.feats for word in words),
        "dependencies": bool(words) and all(word.deprel for word in words),
        "ner": "ner" in pipeline.processors and bool(all_entities),
        **candidate_capabilities,
    }
    return {
        "backend": "Stanza",
        "model": "default",
        "capabilities": {name: _status(value) for name, value in capability.items()},
        "assurance": "TESTED",
        "fixture_count": len(docs),
        "evidence": {
            "sentence_count": sum(len(doc.sentences) for doc in docs),
            "token_count": len(words),
            "morphology_feature_token_count": sum(bool(word.feats) for word in words),
            "ner_entity_count": len(all_entities),
            "ner_actor_entity_count": len(actor_entities),
            **candidate_evidence,
        },
    }


def main() -> None:
    report = {
        "scope": "Technical fixture feasibility only; no CDA skill is methodologically validated.",
        "en": _spacy("en", "en_core_web_sm"),
        "es": _spacy("es", "es_core_news_sm"),
        "ar": _stanza_arabic(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
