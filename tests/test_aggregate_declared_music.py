from engine.signals.aggregate import aggregate_declared_music
from engine.signals.result_types import CoverageSummary
from engine.signals.topic_sufficiency import TopicDecision
from engine.standards import load_taxonomy


def _decision(selected):
    return TopicDecision(
        policy_version="english-audio-aggregation-v3", outcome="eligible",
        primary_category=selected[0][0], transcript_seconds=20.0, audio_seconds=30.0,
        primary_role_support_seconds=(selected[0],), secondary_role_support_seconds=(),
        topic_structure="parent_rollup", selected_support_seconds=selected,
    )


def test_rolled_up_entertainment_parent_does_not_restate_declared_music():
    result = aggregate_declared_music(
        (), decision=_decision((("JLBCU7", 15.0),)),
        coverage=CoverageSummary(source_seconds=30.0, analysed_seconds=20.0),
        taxonomy=load_taxonomy(), video=True,
    )
    assert (result.primary_category, result.secondary_categories) == ("338", ("650",))
    assert result.selected_support_seconds == () and result.sentiment is None
