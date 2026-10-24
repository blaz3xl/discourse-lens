import json

from discourselens import Project
from discourselens.reporting import (
    analysis_payload,
    finding_payload,
    render_analysis_html,
    render_analysis_markdown,
    render_finding_html,
    render_finding_markdown,
    write_analysis_report,
)
from discourselens.types import EvidenceRole


def test_finding_and_analysis_reports_keep_observation_interpretation_and_evidence_distinct(
    tmp_path,
):
    project = Project.init(tmp_path / "project")
    source_text = "Support evidence phrase.\nCounter evidence phrase."
    _, revision = project.import_text("speech.txt", source_text)
    run_id = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
    finding_id = project.create_finding(
        run_id,
        "fixture",
        "1.0.0",
        "agency in the clause",
        "OBSERVATION The passive clause omits the actor",
        interpretation="INTERPRETATION The omission backgrounds institutional agency",
        alternative_interpretation="ALTERNATIVE The wording may follow a conventional style",
    )
    support_start = source_text.index("Support")
    counter_start = source_text.index("Counter")
    support_id = project.textual_evidence(
        project.create_span(revision, support_start, support_start + len("Support evidence phrase."))
    )
    counter_id = project.textual_evidence(
        project.create_span(revision, counter_start, counter_start + len("Counter evidence phrase."))
    )
    project.attach_evidence(finding_id, support_id, EvidenceRole.SUPPORT)
    project.attach_evidence(finding_id, counter_id, EvidenceRole.COUNTEREVIDENCE)

    expected_evidence = [
        (support_id, "SUPPORT", "Support evidence phrase.", "[speech.txt · ¶1]"),
        (counter_id, "COUNTEREVIDENCE", "Counter evidence phrase.", "[speech.txt · ¶2]"),
    ]
    finding_json = finding_payload(project, finding_id)
    analysis_json = analysis_payload(project, run_id)
    exported_json_path = write_analysis_report(
        project, run_id, tmp_path / "analysis.json", "json"
    )
    exported_json = json.loads(exported_json_path.read_text(encoding="utf-8"))

    for finding in (
        finding_json["finding"],
        analysis_json["findings"][0]["finding"],
        exported_json["findings"][0]["finding"],
    ):
        assert finding["observation"] == "OBSERVATION The passive clause omits the actor"
        assert finding["interpretation"] == (
            "INTERPRETATION The omission backgrounds institutional agency"
        )
        assert finding["alternative_interpretation"] == (
            "ALTERNATIVE The wording may follow a conventional style"
        )

    for payload in (finding_json, analysis_json["findings"][0], exported_json["findings"][0]):
        assert [
            (item["evidence_id"], item["role"], item["text"], item["citation"])
            for item in payload["evidence"]
        ] == expected_evidence
    repeated = finding_payload(project, finding_id)
    assert [item["citation"] for item in repeated["evidence"]] == [
        item[3] for item in expected_evidence
    ]

    finding_markdown = render_finding_markdown(project, finding_id)
    analysis_markdown = render_analysis_markdown(project, run_id)
    finding_html = render_finding_html(project, finding_id)
    analysis_html = render_analysis_html(project, run_id)

    assert "**Observation:** OBSERVATION The passive clause omits the actor" in finding_markdown
    assert "**Interpretation:** INTERPRETATION The omission backgrounds institutional agency" in (
        finding_markdown
    )
    assert "Observation: OBSERVATION The passive clause omits the actor" in analysis_markdown
    assert "Interpretation: INTERPRETATION The omission backgrounds institutional agency" in (
        analysis_markdown
    )
    assert "<strong>Observation:</strong> OBSERVATION The passive clause omits the actor" in (
        finding_html
    )
    assert "<strong>Interpretation:</strong> INTERPRETATION The omission backgrounds institutional agency" in (
        finding_html
    )
    assert "<strong>Observation:</strong> OBSERVATION The passive clause omits the actor" in (
        analysis_html
    )
    assert "<strong>Interpretation:</strong> INTERPRETATION The omission backgrounds institutional agency" in (
        analysis_html
    )
    for rendered in (finding_markdown, analysis_markdown, finding_html, analysis_html):
        assert "Support evidence phrase" in rendered
        assert "Counter evidence phrase" in rendered
        assert "Support" in rendered
        assert "Counterevidence" in rendered
    for rendered in (finding_markdown, analysis_markdown):
        assert r"\[speech\.txt · ¶1\]" in rendered
        assert r"\[speech\.txt · ¶2\]" in rendered
    for rendered in (finding_html, analysis_html):
        assert "[speech.txt · ¶1]" in rendered
        assert "[speech.txt · ¶2]" in rendered
    escaped_ids = [evidence_id.replace("-", "\\-") for evidence_id in (support_id, counter_id)]
    for rendered in (finding_markdown, analysis_markdown):
        assert all(f"evidence ID: {evidence_id}" in rendered for evidence_id in escaped_ids)
    for rendered in (finding_html, analysis_html):
        assert all(
            f"evidence ID: {evidence_id}" in rendered for evidence_id in (support_id, counter_id)
        )
    assert [item["evidence_id"] for item in exported_json["findings"][0]["evidence"]] == [
        support_id,
        counter_id,
    ]
