import re
from typing import Any
from interfaces import (
    Question, Verifier, LLMTrace, VerifierResponse,
    TrivialVerifierConfig, LLMVerifierConfig
)
from call_llm import call_llm
from utils import format_question

class TrivialVerifier(Verifier):
    """
    Concrete Verifier implementation that always accepts candidate questions trivially.
    """
    def __init__(self, config: TrivialVerifierConfig = None):
        self.config = config or TrivialVerifierConfig()

    def verify(self, candidate: Question) -> VerifierResponse:
        # Enforce strict ASCII on question text to prevent non-ASCII symbols like →
        if not candidate.question_text.isascii():
            print(f"  [Verifier] ❌ Rejected question (non-ASCII question text): {repr(candidate.question_text)}")
            return VerifierResponse(success=False, question=candidate, trace=None)
            
        # Enforce strict ASCII on option texts
        for opt in candidate.options:
            if not opt.isascii():
                print(f"  [Verifier] ❌ Rejected question (non-ASCII option text): {repr(opt)}")
                return VerifierResponse(success=False, question=candidate, trace=None)
                
        # Ensure correct answer matches one of the options (by string value or index letter)
        # Typically, correct_answer can be a letter ('A', 'B', 'C', 'D') or the exact text.
        expected_ans = candidate.correct_answer.strip()
        valid_letters = [chr(x) for x in range(ord('A'), ord('A') + len(candidate.options))]
        if expected_ans not in valid_letters and expected_ans not in candidate.options:
            print(f"  [Verifier] ❌ Rejected question: correct answer '{expected_ans}' is not in choices.")
            return VerifierResponse(success=False, question=candidate, trace=None)
            
        return VerifierResponse(success=True, question=candidate, trace=None)

class LLMVerifier(Verifier):
    """
    Concrete Verifier implementation that uses a zero-shot LLM solver model
    configured by LLMVerifierConfig to verify generated questions.
    """
    def __init__(self, config: LLMVerifierConfig):
        self.config = config

    def verify(self, candidate: Question) -> VerifierResponse:
        system_prompt = """You are an expert in science and math. Your job is to solve the user's multiple choice questions.
You must solve the problem carefully step-by-step, and then output the single uppercase correct option choice letter (e.g. A, B, C, D, E, or F) wrapped inside <answer>...</answer> XML tags.

Example format:
Step-by-step derivation...
Therefore, the correct choice is <answer>C</answer>."""

        question_str = format_question(
            candidate,
            include_answer=False,
            extra_options=["None of the above", "Cannot be determined"]
        )

        user_prompt = f"""Solve the following question:

{question_str}

Derive the solution and output the correct choice letter inside <answer>...</answer> tags."""

        try:
            # call_llm now returns an LLMTrace object directly, using config values
            trace = call_llm(
                model=self.config.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.config.temperature,
                thinking_budget=self.config.thinking_budget,
                max_tokens=self.config.max_tokens
            )
            response = trace.raw_output
            
            success = False
            match = re.search(r"<answer>\s*([A-Z])\s*</answer>", response)
            if match:
                predicted_choice = match.group(1).strip()
                expected = candidate.correct_answer.strip().upper()
                success = (predicted_choice == expected)
                
            return VerifierResponse(success=success, question=candidate, trace=trace)
            
        except Exception as e:
            print(f"Verifier error during independent call: {e}")
            fallback_trace = LLMTrace(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                thinking=None,
                raw_output=str(e)
            )
            return VerifierResponse(success=False, question=candidate, trace=fallback_trace)
