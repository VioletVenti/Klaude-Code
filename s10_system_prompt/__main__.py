"""Interactive entry: `python -m s10_system_prompt`. Reads prompts until q/EOF."""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .agent import AgentState, make_client, run_agent


def main() -> None:
    print("s10 system prompt — type a task, q to quit")
    client = make_client()
    state = AgentState()

    while True:
        try:
            prompt = input("s10 >> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in ("q", "exit"):
            break

        run_agent(prompt, state, client)
        last = state.messages[-1]
        if last["role"] == "assistant":
            print(last["content"], "\n")


if __name__ == "__main__":
    main()
