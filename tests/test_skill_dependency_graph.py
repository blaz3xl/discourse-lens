import pytest

from discourselens.errors import SkillManifestError
from discourselens.skills import SkillDefinition, SkillRegistry
from discourselens.types import ExecutionKind, SkillManifest


def _noop(*_args: object, **_kwargs: object) -> object:
    return None


def _definition(skill_id: str, dependencies: tuple[str, ...] = ()) -> SkillDefinition:
    return SkillDefinition(
        SkillManifest(
            id=skill_id,
            version="1.0.0",
            framework="test",
            execution_kind=ExecutionKind.DETERMINISTIC,
            input_schema_ref="schemas/test-input.json",
            output_schema_ref="schemas/test-output.json",
            executor=f"builtin.{skill_id}",
            methodology_refs=["docs/methodology/test.md"],
            retrieval_plan_ref="retrieval/test",
            dependencies=list(dependencies),
            documentation_ref="docs/test.md",
        ),
        _noop,
    )


def test_plan_includes_transitive_dependencies_registered_later() -> None:
    registry = SkillRegistry()
    registry.register(_definition("report", ("review",)))
    registry.register(_definition("review", ("extract",)))
    registry.register(_definition("extract"))

    plan = registry.plan_dependencies(["report"])

    assert plan.closure == ("extract", "review", "report")
    assert plan.layers == (("extract",), ("review",), ("report",))


def test_plan_groups_independent_skills_into_topological_waves() -> None:
    registry = SkillRegistry()
    registry.register(_definition("a"))
    registry.register(_definition("b"))
    registry.register(_definition("c", ("a",)))
    registry.register(_definition("d", ("b",)))
    registry.register(_definition("e", ("c", "d")))

    plan = registry.plan_dependencies(["e"])

    assert plan.layers == (("a", "b"), ("c", "d"), ("e",))
    assert plan.closure == ("a", "b", "c", "d", "e")


def test_plan_order_is_stable_across_registration_and_request_order() -> None:
    registry = SkillRegistry()
    registry.register(_definition("z-root", ("b", "a")))
    registry.register(_definition("b"))
    registry.register(_definition("a"))
    registry.register(_definition("x"))

    first = registry.plan_dependencies(["x", "z-root"])
    second = registry.plan_dependencies(["z-root", "x"])

    assert first == second
    assert first.layers == (("a", "b", "x"), ("z-root",))


def test_plan_rejects_an_unregistered_dependency_after_registration_finishes() -> None:
    registry = SkillRegistry()
    registry.register(_definition("report", ("missing",)))

    with pytest.raises(SkillManifestError, match="depends on unknown built-in skill 'missing'"):
        registry.plan_dependencies(["report"])


def test_plan_rejects_unknown_requested_skill() -> None:
    registry = SkillRegistry()

    with pytest.raises(SkillManifestError, match="unknown built-in skill 'missing'"):
        registry.plan_dependencies(["missing"])


def test_plan_rejects_cycles_in_requested_dependency_closure() -> None:
    registry = SkillRegistry()
    registry.register(_definition("first", ("second",)))
    registry.register(_definition("second", ("first",)))

    with pytest.raises(SkillManifestError, match="dependency cycle detected"):
        registry.plan_dependencies(["first"])
