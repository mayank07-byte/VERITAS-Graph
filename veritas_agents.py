import os
import sys
from pathlib import Path
from crewai import Agent, Task, Crew, Process, LLM

# Set dummy key so CrewAI's native client connects cleanly to local Ollama
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:11434/v1"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from veritas_core import VeritasGraphEngine

# Connect CrewAI to local IBM Granite 3.0 Small Language Model via Ollama's /v1 API
granite_slm = LLM(
    model="openai/granite3-dense:2b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

def run_veritas_governance_crew(old_fact: str, new_fact: str, nli_confidence: float):
    auditor_agent = Agent(
        role="Temporal Provenance Auditor",
        goal="Analyze conflicting enterprise facts and explain the exact policy or data drift.",
        backstory="You are an enterprise AI governance auditor inside VERITAS-Graph.",
        llm=granite_slm,
        verbose=True
    )

    guardrail_agent = Agent(
        role="Rogue-Agent Safety & Repair Controller",
        goal="Prevent unauthorized autonomous memory overwrites when NLI confidence is ambiguous.",
        backstory="You enforce deterministic safety guardrails so AI agents never go rogue.",
        llm=granite_slm,
        verbose=True
    )

    audit_task = Task(
        description=(
            f"Compare Old Memory: '{old_fact}' with New Incoming Memory: '{new_fact}'. "
            f"PyTorch DeBERTa NLI Contradiction Confidence is {nli_confidence:.2f}. "
            "Briefly state what changed in 2 bullet points."
        ),
        expected_output="2 concise bullet points explaining the factual conflict.",
        agent=auditor_agent
    )

    repair_task = Task(
        description=(
            "Based on the conflict audit, output a JSON governance decision with keys: "
            "'action' ('DEPRECATE_OLD_NODE' or 'BLOCK_ROGUE_MUTATION_AND_ESCALATE'), "
            "'verified_fact', and 'safety_rationale'."
        ),
        expected_output="A clean JSON governance decision.",
        agent=guardrail_agent
    )

    crew = Crew(
        agents=[auditor_agent, guardrail_agent],
        tasks=[audit_task, repair_task],
        process=Process.sequential
    )
    return crew.kickoff()

if __name__ == "__main__":
    print("\n[VERITAS-Graph] Launching CrewAI Multi-Agent Governance with IBM Granite-3.0 (2B)...")
    decision = run_veritas_governance_crew(
        old_fact="Cloud GPU quota is capped at 4 A100 nodes per research team.",
        new_fact="Research teams may request up to 16 H100 GPUs with lab director sign-off.",
        nli_confidence=0.72
    )
    print("\n================ FINAL CREWAI + GRANITE GOVERNANCE OUTPUT ================")
    print(decision)
