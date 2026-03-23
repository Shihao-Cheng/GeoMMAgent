# exec_agents/reasoning/mcq_match_prompt.py
# Shared MCQ template and option parsing for Reasoning and evaluation tooling.

import re

MCQ_MATCH_TEMPLATE_RAW = (
    "You are an AI assistant who will help me to match an answer "
    "with several options of a single-choice question. "
    "You are provided with a question, several options, and an answer, "
    "and you need to find which option is most similar to the answer. "
    "If the meaning of all options are significantly different "
    "from the answer, output E. "
    "Your should output a single uppercase character in A, B, C, D "
    "(if they are valid options), and E. \n"
    "Example 1: \n"
    "Question: What is the main object in image?\nOptions: A. teddy bear "
    "B. rabbit C. cat D. dog\nAnswer: a cute teddy bear\nYour output: A\n"
    "Example 2: \n"
    "Question: What is the main object in image?\nOptions: A. teddy bear "
    "B. rabbit C. cat D. dog\nAnswer: Spider\nYour output: E\n"
    "Example 3: \n"
    "Question: {}?\nOptions: {}\nAnswer: {}\nYour output: "
)

def parse_options(options_part: str) -> str:
    """Format the ``A:`` … ``D:`` option block into a single spaced string."""
    option_labels = ["A:", "B:", "C:", "D:"]
    option_positions = []
    for label in option_labels:
        pos = options_part.find(label)
        if pos != -1:
            option_positions.append((pos, label))

    option_positions.sort(key=lambda x: x[0])

    extracted_options = []
    for i, (pos, label) in enumerate(option_positions):
        start = pos + len(label)
        if i + 1 < len(option_positions):
            end = option_positions[i + 1][0]
            option_content = options_part[start:end].strip()
        else:
            option_content = options_part[start:].strip()

        option_content = re.sub(r"\s+", " ", option_content).strip()
        if option_content:
            extracted_options.append(f"{label.strip(':')} {option_content}")

    if not extracted_options:
        return "A B C D"

    return " ".join(extracted_options)


def build_mcq_match_prompt(question: str, options: str, prediction: str) -> str:
    """Fill the MCQ template with question stem, formatted options, and reference answer text."""
    return MCQ_MATCH_TEMPLATE_RAW.format(question.strip("?"), options, prediction)


def build_reasoning_user_message(original_task: str, answer_text: str) -> str:
    """
    Build the Reasoning user message: split ``original_task`` on ``?``, parse options, then
    call :func:`build_mcq_match_prompt`.

    ``original_task`` must place ``?`` before the option block; the block must contain
    ``A:`` … ``D:`` labels so :func:`parse_options` can read them.

    ``answer_text`` is the reference answer string from the dataset (e.g. evaluation parquet
    ``answer``). When omitted upstream, callers may pass an empty string for interactive-only runs.
    """
    parts = original_task.split("?")
    if len(parts) >= 2:
        question = parts[0] + "?"
        options_part = "?".join(parts[1:]).strip()
    else:
        question = original_task
        options_part = ""

    options_str = parse_options(options_part)
    return build_mcq_match_prompt(question, options_str, answer_text)


# Reasoning uses no extra system prompt; the task is fully specified in the user message.
REASONING_SYSTEM_PROMPT = ""
