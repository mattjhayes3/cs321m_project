from typing import List, Optional
from interfaces import (
    Question, Benchmark, TargetProfile, QuestionGeneratorConfig,
    GenerationResponse, Prompter, Verifier, TargetSelector,
    VerifierConfig, PrompterConfig, TargetSelectorConfig,
    VerifierResponse
)
from verifier import TrivialVerifier, LLMVerifier
from prompter import NearbyExamplePrompter, ScaledExamplePrompter, IncreaseDifficultyPrompter, AddOptionPrompter
from target_selector import MidpointTargetSelector

def build_verifier(config: VerifierConfig) -> Verifier:
    """
    Factory builder that constructs the concrete Verifier based on configuration type.
    """
    if config.type == "trivial_verifier":
        return TrivialVerifier(config)
    elif config.type == "llm_verifier":
        return LLMVerifier(config)
    else:
        raise ValueError(f"Unsupported verifier config type: '{config.type}'")

def build_prompter(config: PrompterConfig) -> Prompter:
    """
    Factory builder that constructs the concrete Prompter/Exemplar-Selector.
    """
    if config.type == "nearby_example_prompter":
        return NearbyExamplePrompter(config)
    elif config.type == "scaled_example_prompter":
        return ScaledExamplePrompter(config)
    elif config.type == "increase_difficulty_prompter":
        return IncreaseDifficultyPrompter(config)
    elif config.type == "add_option_prompter":
        return AddOptionPrompter(config)
    else:
        raise ValueError(f"Unsupported prompter config type: '{config.type}'")

def build_target_selector(config: TargetSelectorConfig) -> TargetSelector:
    """
    Factory builder that constructs the concrete TargetSelector.
    """
    if config.type == "midpoint_target_selector":
        return MidpointTargetSelector(config)
    else:
        raise ValueError(f"Unsupported target selector config type: '{config.type}'")

class QuestionGenerator:
    """
    Concrete Question Generator that dynamically constructs its verifiers,
    prompters, and target selectors from configuration parameters, executing the active synthesis loop.
    """
    def __init__(self, config: QuestionGeneratorConfig):
        self.config = config
        self.verifier = build_verifier(config.verifier)
        self.prompter = build_prompter(config.prompter)
        self.target_selector = build_target_selector(config.target_selector)

    def generate(self, benchmark: Benchmark, exclude_exemplar_ids: Optional[List[str]] = None) -> GenerationResponse:
        # 1. Select optimal target difficulty midpoint (or other custom loading profiles)
        target_profile = self.target_selector.select_target(benchmark)

        # 2. Retrieve exemplars, call LLM with reasoning, and parse generated candidates
        prompter_response = self.prompter.get_examples(benchmark, target_profile, exclude_ids=exclude_exemplar_ids)

        # 3. Run solvability checks on all generated candidates
        verifier_responses = [self.verifier.verify(q) for q in prompter_response.questions]

        # 4. Collect verified successful questions and compile responses
        verified_questions = [vr.question for vr in verifier_responses if vr.success]

        return GenerationResponse(
            verified_questions=verified_questions,
            target_profile=target_profile,
            prompter_response=prompter_response,
            verifier_responses=verifier_responses
        )
