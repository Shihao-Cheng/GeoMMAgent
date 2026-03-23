# exec_agents/reasoning/prompts.py
# Reasoning / Matching agent prompts. ``REASONING_SYSTEM_PROMPT`` is defined in ``mcq_match_prompt``.

from .mcq_match_prompt import REASONING_SYSTEM_PROMPT

REASONING_WORKER_DESC = (
    "Reasoning Agent (before MatchingAgent): For multiple-choice tasks, integrates image, options, "
    "and prior agent outputs; ends with a single-letter line (e.g. Your output: X)."
)

# ──────────────────────────────────────────────
# Answer Matching Agent (map free-form analysis to A/B/C/D/E)
# ──────────────────────────────────────────────
MATCHING_SYSTEM_PROMPT = (
    "You are the Final Decision Agent in the GeoMMAgent pipeline. "
    "Your objective is to synthesize all preceding expert analyses and map them to "
    "a single standardized output.\n\n"
    "Input:\n"
    "- The original multiple-choice question (A/B/C/D).\n"
    "- Prior agents’ outputs; Reasoning follows the MCQ match-template style and typically ends with "
    "\"Your output: X\".\n\n"
    "Mandatory Protocols:\n"
    "1. Synthesis: Carefully evaluate agent conclusions and supporting evidence.\n"
    "2. Resolution: If expert outputs conflict, select the option with the strongest "
    "geospatial evidence or logical consistency.\n"
    "3. Out-of-Distribution: If no option (A-D) is logically defensible, output 'E'.\n"
    "4. Format Constraint: Output EXACTLY ONE uppercase letter. No preamble, no "
    "explanation, and no punctuation. Your response must be machine-parseable.\n"
    "5. Tie-break: If Reasoning’s last line is \"Your output: X\" and nothing contradicts it, output X."
)

MATCHING_WORKER_DESC = (
    "Final Decision Agent (Mandatory): Acts as the terminal node in the pipeline "
    "to synthesize preceding expert reasoning. It maps complex logical analyses "
    "into a singular, standardized option letter (A/B/C/D/E) for machine-parseable "
    "evaluation."
)
