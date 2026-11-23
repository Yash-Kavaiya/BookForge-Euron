"""Agent-graph wiring: the whole tree constructs and has the right shape."""

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent

from bookforge.agents.analyst import reuse_cached_analysis
from bookforge.agents.critic import skip_critic_unless_machine_ok
from bookforge.agents.orchestrator import (
    BookProductionAgent,
    build_chapter_pipeline,
    build_root_agent,
)
from bookforge.agents.verifier import ChapterVerifierAgent
from bookforge.agents.writer import gate_refiner_on_critique, reuse_written_chapter
from bookforge.config import Settings, get_settings
from bookforge.instrumentation import (
    after_agent_timer,
    after_model_usage,
    before_agent_timer,
    before_model_usage,
)
from bookforge.schemas import CritiqueReport


def test_root_graph_shape():
    root = build_root_agent(get_settings())
    assert isinstance(root, SequentialAgent)
    names = [a.name for a in root.sub_agents]
    assert names == ["channel_intake", "book_production", "book_compiler"]


def test_chapter_pipeline_shape():
    pipeline = build_chapter_pipeline(get_settings())
    names = [a.name for a in pipeline.sub_agents]
    assert names == [
        "media_acquisition",
        "transcript_analyst",
        "visual_assets",
        "chapter_writer",
        "chapter_qa_loop",
    ]
    qa = pipeline.sub_agents[-1]
    assert isinstance(qa, LoopAgent)
    assert qa.max_iterations == get_settings().qa_max_iterations
    # deterministic gate first, LLM critic second, refiner last
    assert [a.name for a in qa.sub_agents] == [
        "chapter_verifier",
        "chapter_critic",
        "chapter_refiner",
    ]
    assert isinstance(qa.sub_agents[0], ChapterVerifierAgent)
    assert not isinstance(qa.sub_agents[0], LlmAgent)


def test_llm_agents_are_leaf_workers():
    pipeline = build_chapter_pipeline(get_settings())
    qa = pipeline.sub_agents[-1]
    llms = [a for a in list(pipeline.sub_agents) + list(qa.sub_agents) if isinstance(a, LlmAgent)]
    assert {a.name for a in llms} == {
        "transcript_analyst",
        "chapter_writer",
        "chapter_critic",
        "chapter_refiner",
    }
    for agent in llms:
        assert agent.disallow_transfer_to_parent is True
        assert not agent.sub_agents  # no transfer targets


def test_production_agent_registers_pipeline_as_subagent():
    pipeline = build_chapter_pipeline(get_settings())
    production = BookProductionAgent(pipeline=pipeline)
    assert production.pipeline is pipeline
    assert production.sub_agents == [pipeline]


def test_analyst_uses_structured_output_and_cache_callback():
    pipeline = build_chapter_pipeline(get_settings())
    analyst = pipeline.sub_agents[1]
    assert isinstance(analyst, LlmAgent)
    assert analyst.output_schema is not None
    assert analyst.output_key == "analysis_json"
    assert not analyst.tools  # output_schema forbids tools; analyst has none
    assert analyst.before_agent_callback[0] is reuse_cached_analysis


def test_critic_is_structured_and_gated():
    pipeline = build_chapter_pipeline(get_settings())
    critic = pipeline.sub_agents[-1].sub_agents[1]
    assert isinstance(critic, LlmAgent)
    assert critic.output_schema is CritiqueReport
    assert critic.output_key == "critique_report"
    assert not critic.tools  # no free-form tool protocol any more
    assert critic.before_agent_callback[0] is skip_critic_unless_machine_ok


def test_writer_and_refiner_callbacks():
    pipeline = build_chapter_pipeline(get_settings())
    writer = pipeline.sub_agents[3]
    refiner = pipeline.sub_agents[-1].sub_agents[2]
    assert writer.before_agent_callback[0] is reuse_written_chapter
    assert refiner.before_agent_callback[0] is gate_refiner_on_critique
    assert writer.output_key == refiner.output_key == "chapter_tex"


def test_every_stage_is_instrumented():
    pipeline = build_chapter_pipeline(get_settings())
    stages: list[BaseAgent] = list(pipeline.sub_agents[:-1]) + list(
        pipeline.sub_agents[-1].sub_agents
    )
    for agent in stages:
        assert before_agent_timer in agent.before_agent_callback, agent.name
        assert after_agent_timer in agent.after_agent_callback, agent.name
        if isinstance(agent, LlmAgent):
            assert before_model_usage in agent.before_model_callback, agent.name
            assert after_model_usage in agent.after_model_callback, agent.name


def test_per_stage_model_routing(tmp_path):
    settings = Settings(
        _env_file=None,
        compile_latex=False,
        workspace_root=tmp_path,
        analyst_model="gemini-2.5-flash",
        writer_model="gemini-2.5-pro",
        critic_model="gemini-2.5-flash-lite",
        refiner_model="",
        openai_api_key="",
    )
    pipeline = build_chapter_pipeline(settings)
    analyst, writer = pipeline.sub_agents[1], pipeline.sub_agents[3]
    critic, refiner = pipeline.sub_agents[-1].sub_agents[1:]
    assert analyst.model == "gemini-2.5-flash"
    assert writer.model == "gemini-2.5-pro"
    assert critic.model == "gemini-2.5-flash-lite"
    assert refiner.model == "gemini-2.5-pro"  # falls back to the writer model
    assert analyst.generate_content_config.temperature == settings.analyst_temperature
    assert critic.generate_content_config.temperature == settings.critic_temperature

    settings.refiner_model = "gemini-2.5-flash"
    refiner = build_chapter_pipeline(settings).sub_agents[-1].sub_agents[2]
    assert refiner.model == "gemini-2.5-flash"
