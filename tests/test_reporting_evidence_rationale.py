from discourselens import Project
from discourselens.reporting import (
    analysis_payload,
    finding_payload,
    render_analysis_html,
    render_analysis_markdown,
    render_finding_html,
    render_finding_markdown,
)
from discourselens.types import EvidenceRole


def test_finding_evidence_rationale_survives_payload_and_escaped_reports(tmp_path):
    rationale = "Supports agency reading.\n<script>alert(9)</script>"
    with Project.init(tmp_path / "evidence-rationale") as project:
        _, revision = project.import_text("speech.txt", "Officials approved the plan.")
        run_id = project.create_run(project.snapshot([revision]), {})
        finding_id = project.create_finding(
            run_id,
            "fixture",
            "1.0.0",
            "Agency construction",
            "Officials are represented as agents.",
        )
        span_id = project.create_span(revision, 0, len("Officials approved the plan."))
        evidence_id = project.textual_evidence(span_id)
        project.attach_evidence(
            finding_id, evidence_id, EvidenceRole.SUPPORT, rationale=rationale
        )

        finding = finding_payload(project, finding_id)
        analysis = analysis_payload(project, run_id)
        assert finding["evidence"][0]["rationale"] == rationale
        assert analysis["findings"][0]["evidence"][0]["rationale"] == rationale

        finding_markdown = render_finding_markdown(project, finding_id)
        analysis_markdown = render_analysis_markdown(project, run_id)
        finding_html = render_finding_html(project, finding_id)
        analysis_html = render_analysis_html(project, run_id)

        for rendered in (finding_markdown, analysis_markdown):
            assert r"Link rationale: Supports agency reading\." in rendered
            assert r"\<script\>alert\(9\)\<\/script\>" in rendered
            assert "\n<script>" not in rendered
        for rendered in (finding_html, analysis_html):
            assert "Link rationale:</strong> Supports agency reading." in rendered
            assert "&lt;script&gt;alert(9)&lt;/script&gt;" in rendered
            assert "<script>alert(9)</script>" not in rendered


def test_reports_omit_empty_finding_evidence_rationale(tmp_path):
    with Project.init(tmp_path / "no-evidence-rationale") as project:
        _, revision = project.import_text("speech.txt", "Officials approved the plan.")
        run_id = project.create_run(project.snapshot([revision]), {})
        finding_id = project.create_finding(
            run_id, "fixture", "1.0.0", "Agency construction", "An observation."
        )
        evidence_id = project.textual_evidence(
            project.create_span(revision, 0, len("Officials approved the plan."))
        )
        project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)

        assert finding_payload(project, finding_id)["evidence"][0]["rationale"] is None
        assert "Link rationale:" not in render_finding_markdown(project, finding_id)
        assert "Link rationale:" not in render_finding_html(project, finding_id)
