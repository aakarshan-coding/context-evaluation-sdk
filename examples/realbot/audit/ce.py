"""Run the contexteval CLI with every OpenAI request's usage tallied. Usage: python ce.py <args...>"""
import sys
import tally

tally.COMMAND[0] = sys.argv[1] if len(sys.argv) > 1 else "?"
from contexteval._cli import main  # noqa: E402

sys.exit(main(sys.argv[1:]))
