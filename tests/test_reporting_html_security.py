from discourselens import Project
from discourselens.reporting import render_analysis_html, render_finding_html
from discourselens.types import EvidenceRole


def test_html_reports_escape_untrusted_corpus_metadata_findings_and_citations(tmp_path):
    corpus_markup = "<img src=x onerror=alert(1)>"
    metadata_markup = "<svg onload=alert(2)>"
    finding_markup = "<script>alert(3)</script>"
    citation_markup = 'source\"><img src=x onerror=alert(4)>.txt'
    review_markup = "<iframe srcdoc=alert(5)>"

    project = Project.init(tmp_path / "project")
    _, revision = project.import_text(citation_markup, f"Passage: {corpus_markup}")
    run = project.create_run(
        project.snapshot([revision]), {"research_label": metadata_markup}
    )
    evidence = project.textual_evidence(project.create_span(revision, 0, len(f"Passage: {corpus_markup}")))
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        finding_markup,
        corpus_markup,
        interpretation=metadata_markup,
    )
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)
    project.review_finding(finding, "MODIFIED", review_markup)

    rendered_reports = (
        render_finding_html(project, finding),
        render_analysis_html(project, run),
    )

    for report in rendered_reports:
        assert "<img src=x" not in report
        assert "<svg onload" not in report
        assert "<script>alert(3)</script>" not in report
        assert "<iframe srcdoc" not in report
        assert "&lt;img src=x onerror=alert(1)&gt;" in report
        assert "&lt;svg onload=alert(2)&gt;" in report
        assert "&lt;script&gt;alert(3)&lt;/script&gt;" in report
        assert "&lt;iframe srcdoc=alert(5)&gt;" in report
        assert "&quot;&gt;&lt;img src=x onerror=alert(4)&gt;" in report
