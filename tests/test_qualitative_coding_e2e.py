import json
import shutil

from discourselens import Project, verify_research_bundle


def test_qualitative_coding_survives_reopen_copy_and_scoped_exports(tmp_path):
    project_root = tmp_path / "research-project"
    copied_root = tmp_path / "portable-copy"
    selected_text = "Café debate: migrant voices describe public policy."

    with Project.init(project_root) as project:
        _, selected_revision = project.import_text("selected.txt", selected_text)
        _, outside_revision = project.import_text(
            "outside.txt", "OUT_OF_SCOPE_SOURCE_SENTINEL describes distant events."
        )
        selected_scope = project.snapshot([selected_revision])
        outside_scope = project.snapshot([outside_revision])

        codebook = project.create_codebook(
            "discourse actors", "Researcher-authored descriptive categories."
        )
        root_code = project.create_code(codebook, "representation strategy")
        parent_code = project.create_code(
            codebook, "social actor", "A represented actor.", parent_code_id=root_code
        )
        child_code = project.create_code(
            codebook,
            "collective actors",
            "A group represented as acting together.",
            parent_code_id=parent_code,
        )
        unrelated_code = project.create_code(codebook, "UNANNOTATED_CODE_SENTINEL")
        outside_codebook = project.create_codebook("OUT_OF_SCOPE_CODEBOOK_SENTINEL")
        outside_code = project.create_code(
            outside_codebook, "OUT_OF_SCOPE_CODE_SENTINEL"
        )

        start = selected_text.index("migrant voices")
        end = start + len("migrant voices")
        span = project.create_span(selected_revision, start, end)
        annotation = project.annotate_span(child_code, span, "Researcher-coded passage.")
        outside_span = project.create_span(outside_revision, 0, 12)
        outside_annotation = project.annotate_span(
            outside_code, outside_span, "OUT_OF_SCOPE_ANNOTATION_SENTINEL"
        )
        memo = project.create_research_memo(
            selected_scope, "Selection rationale", "The selected source frames collective action."
        )
        outside_memo = project.create_research_memo(
            outside_scope,
            "OUT_OF_SCOPE_MEMO_SENTINEL",
            "This memo belongs to another immutable selection.",
        )
        analysis_run = project.create_run(selected_scope, {"profile": "coding-e2e"})
        project.finalize_run(analysis_run)
        project_id = project.project_id

    with Project.open(project_root) as reopened:
        assert reopened.project_id == project_id
        assert reopened.revision_text(selected_revision) == selected_text
        assert reopened.scope_revisions(selected_scope) == [selected_revision]
        assert reopened.span_text(span) == "migrant voices"
        assert reopened.render_span_location(span) == "[selected.txt · ¶1]"

    shutil.copytree(project_root, copied_root)
    with Project.open(copied_root) as portable:
        assert portable.project_id == project_id
        assert portable.revision_text(selected_revision) == selected_text
        assert portable.span_text(span) == "migrant voices"
        assert portable.scope_revisions(selected_scope) == [selected_revision]

        annotations = portable.annotations_for_scope(selected_scope)
        assert [row["id"] for row in annotations] == [annotation]
        assert annotations[0]["code_id"] == child_code
        assert annotations[0]["start_offset"] == start
        assert annotations[0]["end_offset"] == end
        assert [row["id"] for row in portable.research_memos(selected_scope)] == [memo]
        assert portable.annotations_for_scope(outside_scope)[0]["id"] == outside_annotation
        assert [row["id"] for row in portable.research_memos(outside_scope)] == [outside_memo]

        hierarchy = portable.db.execute(
            "SELECT id,parent_code_id FROM codes WHERE id IN (?,?,?) ORDER BY id",
            (root_code, parent_code, child_code),
        ).fetchall()
        assert {row["id"] for row in hierarchy} == {root_code, parent_code, child_code}
        parents = {row["id"]: row["parent_code_id"] for row in hierarchy}
        assert parents == {
            root_code: None,
            parent_code: root_code,
            child_code: parent_code,
        }

        report_path = portable.export_analysis_report(
            analysis_run, tmp_path / "analysis.json", format="json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert [row["revision_id"] for row in report["qualitative_coding"]["annotations"]] == [
            selected_revision
        ]
        assert report["qualitative_coding"]["annotations"][0]["text"] == "migrant voices"
        assert report["qualitative_coding"]["annotations"][0]["citation"] == (
            "[selected.txt · ¶1]"
        )
        assert [row["id"] for row in report["qualitative_coding"]["memos"]] == [memo]
        assert [row["id"] for row in report["document_revisions"]] == [selected_revision]

        bundle_path = portable.export_research_bundle(
            analysis_run, tmp_path / "research-bundle"
        )
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        coding = bundle["qualitative_coding"]
        assert [row["id"] for row in coding["annotations"]] == [annotation]
        assert coding["source_spans"] == [
            {
                "id": span,
                "revision_id": selected_revision,
                "start_offset": start,
                "end_offset": end,
            }
        ]
        verification = verify_research_bundle(bundle_path)
        assert verification["status"] == "VALID"
        assert verification["integrity"] == "VERIFIED_SHA256"
        codes_by_id = {row["id"]: row for row in coding["codes"]}
        assert set(codes_by_id) == {root_code, parent_code, child_code}
        assert codes_by_id[root_code]["parent_code_id"] is None
        assert codes_by_id[parent_code]["parent_code_id"] == codes_by_id[root_code]["id"]
        assert codes_by_id[child_code]["parent_code_id"] == codes_by_id[parent_code]["id"]
        assert unrelated_code not in codes_by_id
        assert outside_code not in codes_by_id
        assert [row["id"] for row in coding["codebooks"]] == [codebook]
        assert [row["id"] for row in coding["memos"]] == [memo]
        assert [row["id"] for row in bundle["document_revisions"]] == [selected_revision]
        serialized_exports = json.dumps([report, bundle], ensure_ascii=False)
        for sentinel in (
            "OUT_OF_SCOPE_SOURCE_SENTINEL",
            "OUT_OF_SCOPE_CODEBOOK_SENTINEL",
            "OUT_OF_SCOPE_CODE_SENTINEL",
            "UNANNOTATED_CODE_SENTINEL",
            unrelated_code,
            "OUT_OF_SCOPE_ANNOTATION_SENTINEL",
            "OUT_OF_SCOPE_MEMO_SENTINEL",
            outside_revision,
            outside_annotation,
            outside_memo,
        ):
            assert sentinel not in serialized_exports
