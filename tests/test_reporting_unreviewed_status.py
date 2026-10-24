from discourselens import Project
from discourselens.reporting import (
    finding_payload,
    render_analysis_html,
    render_analysis_markdown,
    render_finding_html,
    render_finding_markdown,
)
from discourselens.types import EvidenceRole


def test_unreviewed_finding_status_is_visible_in_all_report_formats(tmp_path):
    project = Project.init(tmp_path / "unreviewed-report")
    try:
        _, revision_id = project.import_text("speech.txt", "Apples are green.")
        run_id = project.create_run(project.snapshot([revision_id]), {})
        finding_id = project.create_finding(
            run_id,
            "fixture",
            "1.0.0",
            "descriptive observation",
            "The sentence describes apples.",
        )
        evidence_id = project.textual_evidence(project.create_span(revision_id, 0, 6))
        project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)

        payload = finding_payload(project, finding_id)
        assert payload["finding"]["review_status"] == "GENERATED"
        assert payload["human_reviews"] == []

        finding_markdown = render_finding_markdown(project, finding_id)
        analysis_markdown = render_analysis_markdown(project, run_id)
        finding_html = render_finding_html(project, finding_id)
        analysis_html = render_analysis_html(project, run_id)

        assert "**Human review status:** GENERATED" in finding_markdown
        assert "Human review status: GENERATED" in analysis_markdown
        assert "<strong>Human review status:</strong> GENERATED" in finding_html
        assert "<strong>Human review status:</strong> GENERATED" in analysis_html
    finally:
        project.close()
