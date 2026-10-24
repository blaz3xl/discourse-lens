from discourselens import Project
from discourselens.vectors import VectorChunk, build_vector_index, hybrid_search


def test_hybrid_search_uses_literal_bm25_order_not_paragraph_id_order(tmp_path):
    project = Project.init(tmp_path / "project")
    long_text = "green policy " + "context " * 120
    short_text = "green policy"
    nonliteral_text = "green framework followed later by policy"
    _, revision = project.import_text(
        "selected.txt", f"{long_text}\n{short_text}\n{nonliteral_text}"
    )
    _, outside_revision = project.import_text("outside.txt", "green policy")

    paragraphs = project.db.execute(
        "SELECT id,ordinal,start_offset,end_offset,text FROM paragraphs "
        "WHERE revision_id=? ORDER BY ordinal",
        (revision,),
    ).fetchall()
    selected_ids = {
        0: ("a-long", (0.2, 0.0)),
        1: ("z-short", (0.9, 0.0)),
        2: ("m-nonliteral", (0.5, 0.0)),
    }
    chunks: list[VectorChunk] = []
    for paragraph in paragraphs:
        paragraph_id, embedding = selected_ids[paragraph["ordinal"]]
        previous_id = str(paragraph["id"])
        project.db.execute("UPDATE paragraphs SET id=? WHERE id=?", (paragraph_id, previous_id))
        project.db.execute(
            "UPDATE revision_fts SET paragraph_id=? WHERE paragraph_id=?",
            (paragraph_id, previous_id),
        )
        if paragraph["ordinal"] < 2:
            chunks.append(
                VectorChunk(
                    paragraph_id,
                    paragraph["start_offset"],
                    paragraph["end_offset"],
                    embedding,
                )
            )
    project.db.commit()

    # Establish the regression: the legacy exact-search/canonical order puts the
    # long paragraph first by ID, while BM25 should favor the concise literal match.
    scope = project.snapshot([revision])
    assert [row["paragraph_id"] for row in project.search_exact("green policy", scope)] == [
        "a-long",
        "z-short",
    ]
    ranked = project.search_exact_ranked("green policy", scope)
    assert [row["paragraph_id"] for row in ranked] == ["z-short", "a-long"]
    assert outside_revision not in {row["revision_id"] for row in ranked}
    assert all(row["paragraph_id"] != "m-nonliteral" for row in ranked)

    artifact = build_vector_index(project, revision, chunks)
    hits = hybrid_search(project, scope, [artifact], "green policy", (1.0, 0.0), limit=3)

    assert [(hit.unit_id, hit.lexical_rank, hit.vector_rank) for hit in hits] == [
        ("z-short", 1, 1),
        ("a-long", 2, 2),
    ]
    assert all(hit.revision_id == revision for hit in hits)
