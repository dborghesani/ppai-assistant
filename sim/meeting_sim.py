"""Standalone simulation of a multi-agent work meeting.

The driver is running late for a work meeting, so the in-vehicle assistant
attends on their behalf. A crew of "colleague" agents, coordinated by a
manager agent, discusses a project; the assistant only listens and then
produces a short summary to relay to the driver.

This is a first prototype for multi-agent integration: it runs standalone,
independent of the rest of the assistant pipeline (no telemetry, no TTS).

Run: python sim/meeting_sim.py [--topic "..."] [--ollama-model qwen2.5:3b-instruct]
"""
from __future__ import annotations

import argparse

import structlog
from crewai import LLM, Agent, Crew, Process, Task

logger = structlog.get_logger()

DEFAULT_TOPIC = (
    "Draft the high-level specification for a new autonomous (self-parking) "
    "parking system: scope, key use cases, required sensors/actuators, safety "
    "constraints, and driver-facing behavior."
)


def build_llm(host: str, port: int, model: str, timeout: int, max_tokens: int) -> LLM:
    return LLM(
        model=model,
        base_url=f"http://{host}:{port}",
        timeout=timeout,
        max_tokens=max_tokens,
    )


def build_crew(llm: LLM, topic: str, verbose: bool) -> tuple[Crew, Task]:
    """Build the meeting crew. Returns the crew and the task holding the driver-facing summary."""
    manager = Agent(
        role="Jordan Reyes, Engineering Meeting Manager",
        goal="Keep the meeting focused, make sure every voice is heard, and close with clear decisions.",
        backstory=(
            "You run weekly cross-functional design reviews. You open meetings with a clear "
            "agenda, keep contributions concise and on-topic, and end by summarizing the "
            "decisions and open action items."
        ),
        verbose=verbose,
        llm=llm,
    )
    product_manager = Agent(
        role="Priya Shah, Product Manager",
        goal="Represent user needs and business scope for the project under discussion.",
        backstory=(
            "You define product scope and priorities. You focus on what the feature must "
            "achieve for the end user and where the boundaries of the first release are."
        ),
        verbose=verbose,
        llm=llm,
    )
    systems_engineer = Agent(
        role="Marco Bellini, Systems Engineer",
        goal="Propose a technically sound architecture for the project under discussion.",
        backstory=(
            "You design vehicle software/hardware architectures. You reason about sensors, "
            "actuators, control loops, and integration constraints."
        ),
        verbose=verbose,
        llm=llm,
    )
    safety_engineer = Agent(
        role="Elena Novak, Functional Safety Engineer",
        goal="Surface safety, regulatory and edge-case risks for the project under discussion.",
        backstory=(
            "You review automotive designs for functional safety (ISO 26262) and regulatory "
            "compliance. You raise concrete failure modes and the safeguards they require."
        ),
        verbose=verbose,
        llm=llm,
    )
    ux_designer = Agent(
        role="Sam Whitfield, UX Designer",
        goal="Define how the driver and passengers should experience the project under discussion.",
        backstory=(
            "You design in-vehicle human-machine interaction. You focus on driver trust, "
            "clarity of feedback, and minimizing distraction."
        ),
        verbose=verbose,
        llm=llm,
    )
    driver_delegate = Agent(
        role="Driver's Assistant",
        goal=(
            "Attend the meeting on behalf of the driver, who could not be present, and "
            "report back only what the driver needs to know."
        ),
        backstory=(
            "You are the in-vehicle assistant. The driver is running late and asked you to sit "
            "in on this meeting. You never speak during the discussion; you only listen and "
            "then brief the driver afterwards in plain, spoken language."
        ),
        verbose=verbose,
        llm=llm,
    )

    open_meeting = Task(
        description=(
            f"Open the meeting about the following project:\n{topic}\n\n"
            "State the agenda in 3-5 bullet points: what must be decided today."
        ),
        expected_output="A short agenda for the meeting, as bullet points.",
        agent=manager,
    )
    pm_input = Task(
        description="Present the product scope, target users and priorities for this project.",
        expected_output="A short statement of scope and priorities.",
        agent=product_manager,
    )
    systems_input = Task(
        description="Propose the technical approach: key components, sensors/actuators, and architecture.",
        expected_output="A short technical proposal.",
        agent=systems_engineer,
    )
    safety_input = Task(
        description="Raise the main safety and regulatory risks, and the safeguards they require.",
        expected_output="A short list of risks and required safeguards.",
        agent=safety_engineer,
    )
    ux_input = Task(
        description="Describe how the driver/passengers should experience this feature.",
        expected_output="A short description of the intended driver-facing behavior.",
        agent=ux_designer,
    )
    close_meeting = Task(
        description=(
            "Summarize the discussion into the decisions taken and any open action items. "
            "Be concrete: state what was agreed, not just what was discussed."
        ),
        expected_output="A short list of decisions and open action items.",
        agent=manager,
    )
    driver_summary = Task(
        description=(
            "You missed this meeting. Attendees were Jordan (the manager who ran it), Priya "
            "(product), Marco (systems), Elena (safety) and Sam (UX). Speak directly to the "
            "driver as a natural, friendly debrief right after the meeting ended, based only "
            "on what was actually discussed and decided. Start with something like 'The "
            "meeting just wrapped up' or 'Good news, the meeting is over'. Naturally mention "
            "one or two colleagues by first name and what they said or decided, the way you "
            "would casually recap a meeting to a friend. Do not mention job titles, internal "
            "roles, or that this was an AI-run meeting. Output ONLY the sentences you would "
            "actually speak out loud to the driver \u2014 no preamble, no meta-commentary such "
            "as 'here is the message' or 'this is what you would say'. Keep it to 3-5 short "
            "sentences."
        ),
        expected_output=(
            "The exact spoken sentences for the driver, starting with a natural opener like "
            "'The meeting just wrapped up', naming one or two colleagues, and nothing else."
        ),
        context=[close_meeting],
        agent=driver_delegate,
    )

    crew = Crew(
        agents=[manager, product_manager, systems_engineer, safety_engineer, ux_designer, driver_delegate],
        tasks=[open_meeting, pm_input, systems_input, safety_input, ux_input, close_meeting, driver_summary],
        process=Process.sequential,
        verbose=verbose,
    )
    return crew, driver_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--ollama-host", default="localhost")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--ollama-model", default="qwen2.5:3b-instruct")
    parser.add_argument("--ollama-timeout", type=int, default=1200)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    llm = build_llm(args.ollama_host, args.ollama_port, args.ollama_model, args.ollama_timeout, args.max_tokens)
    crew, driver_summary_task = build_crew(llm, args.topic, args.verbose)

    crew.kickoff()

    print("\n=== Meeting transcript ===\n")
    for task in crew.tasks:
        role = task.agent.role if task.agent else "?"
        output = task.output.raw if task.output else "(no output)"
        print(f"--- {role} ---\n{output}\n")

    print("=== Summary for the driver ===\n")
    print(driver_summary_task.output.raw if driver_summary_task.output else "(no output)")


if __name__ == "__main__":
    main()
