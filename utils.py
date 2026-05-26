from typing import List
from interfaces import Question

def format_option(idx: int, text: str) -> str:
    """
    Formats a multiple-choice option consistently with uppercase letters.
    e.g., format_option(0, "option text") -> "(A) option text"
    """
    return f"({chr(65 + idx)}) {text}"

def format_question(
    question: Question, 
    include_answer: bool = False, 
    extra_options: List[str] = None
) -> str:
    """
    Unified formatting utility to represent a Question object consistently for LLM prompts.
    Represents option choices and the correct answer choice consistently using the format_option helper.
    """
    options = list(question.options)
    if extra_options:
        options.extend(extra_options)

    options_str = "\n".join([
        f"  {format_option(i, opt)}" for i, opt in enumerate(options)
    ])

    blocks = [
        f"Question:\n{question.question_text}",
        f"Options:\n{options_str}"
    ]

    if include_answer:
        ans_key = question.correct_answer.strip().upper()
        if len(ans_key) != 1 or not ("A" <= ans_key <= "Z"):
            raise ValueError(
                f"Invalid correct answer key format in Question. Expected a single uppercase letter (A-Z), "
                f"got: '{ans_key}' (Question ID: {question.id})"
            )
        
        idx = ord(ans_key) - 65
        if 0 <= idx < len(question.options):
            ans_text = question.options[idx]
            blocks.append(f"Correct Answer: {format_option(idx, ans_text)}")
        else:
            raise ValueError(
                f"Correct answer choice letter '{ans_key}' (index {idx}) is out of bounds "
                f"for the options array of length {len(question.options)}: {question.options} "
                f"(Question ID: {question.id})"
            )

    return "\n\n".join(blocks)
