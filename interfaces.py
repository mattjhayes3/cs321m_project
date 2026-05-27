from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

@dataclass
class Factor:
    name: str
    description: str
    loading: float

@dataclass
class Question:
    """
    Represents a single question item in the benchmark pool.
    """
    id: str
    question_text: str
    options: List[str]
    correct_answer: str
    difficulty: float
    discrimination: Optional[float] = None
    factor_loadings: Optional[List[Factor]] = None
    calibrated: bool = False

@dataclass
class TargetProfile:
    """
    Specifies the target properties of the question to generate
    (e.g. target difficulty midpoint beta*, multidimensional loadings V*, and topic constraints).
    """
    target_difficulty: float
    target_factor_loadings: Optional[List[Factor]] = None
    scale: float = 1.0

@dataclass
class SelectionResponse:
    """
    Wrapper containing the selector questions list and the scaling details guiding prompt generation.
    """
    questions: List[Question]
    scaling_metadata: Dict[str, Any] = field(default_factory=dict)

class IRTModel(ABC):
    """
    Abstract Base Class representing a calibrated Item Response Theory model
    (e.g. 1PL/Rasch, 2PL, or Multidimensional IRT).
    """
    @abstractmethod
    def fit(self, response_matrix: pd.DataFrame) -> None:
        """
        Fits the latent traits (theta) and item parameters (difficulty, discrimination)
        using response matrix data.
        """
        pass

    @abstractmethod
    def get_subject_ability(self, subject_id: str) -> List[float]:
        """
        Retrieves the calibrated ability parameters (thetas) for a specific subject model ID.
        """
        pass

    @abstractmethod
    def get_item_information(self, item_id: str, theta: np.ndarray) -> np.ndarray:
        """
        Computes the Fisher Information of a specific item at latent ability theta.
        """
        pass

    @abstractmethod
    def compute_fisher_information(self, theta: np.ndarray) -> np.ndarray:
        """
        Computes the aggregate Fisher Information profile across all calibrated items.
        """
        pass

class Benchmark(ABC):
    """
    Abstract Base Class representing a collection of benchmark questions and their active calibration state.
    """
    def __init__(self, questions: List[Question], irt_model: IRTModel):
        self.questions = questions
        self.calibrated_model = irt_model

    @abstractmethod
    def get_response_matrix(self, model_ids: List[str]) -> pd.DataFrame:
        """
        Compiles binary response data (correct=1, incorrect=0) for evaluated models.
        """
        pass

@dataclass
class LLMTrace:
    """
    Structure encapsulating the full trace of an LLM call for debugging and auditing.
    """
    system_prompt: str
    user_prompt: str
    thinking: Optional[str] = None
    raw_output: str = ""

@dataclass
class PrompterResponse:
    """
    Structure encapsulating the generated candidate questions and the LLM execution trace.
    """
    questions: List[Question]
    trace: LLMTrace
    exemplars: Optional[List[Question]] = None

class Prompter(ABC):
    """
    Abstract Base Class that selects seed exemplars, generates new candidate questions,
    and returns them parsed alongside the LLM execution trace.
    """
    @abstractmethod
    def _select_examples(self, benchmark: Benchmark, target_profile: TargetProfile, exclude_ids: Optional[List[str]] = None) -> List[Question]:
        """
        Selects seed exemplars from the benchmark pool.
        """
        pass

    @abstractmethod
    def get_examples(self, benchmark: Benchmark, target_profile: TargetProfile, exclude_ids: Optional[List[str]] = None) -> PrompterResponse:
        """
        Performs active exemplar selection, prompts LLM, parses, and returns PrompterResponse.
        """
        pass

@dataclass
class VerifierResponse:
    """
    Structure containing the verification success status, verified question, and audit trace.
    """
    success: bool
    question: Question
    trace: Optional[LLMTrace] = None

@dataclass
class GenerationResponse:
    """
    Structure returning the list of verified questions along with full prompter and verifier responses.
    """
    verified_questions: List[Question]
    target_profile: TargetProfile
    prompter_response: PrompterResponse
    verifier_responses: List[VerifierResponse]

class Verifier(ABC):
    """
    Abstract Base Class for programmatic verification and zero-shot solvability checks of generated questions.
    """
    @abstractmethod
    def verify(self, candidate: Question) -> VerifierResponse:
        """
        Verifies candidate syntax, option consistency, and checks solvability to gatekeep quality, returning trace metadata.
        """
        pass

class QuestionGenerator(ABC):
    """
    Abstract Base Class for generating questions given target parameters.
    """
    @abstractmethod
    def generate(self, benchmark: Benchmark) -> GenerationResponse:
        """
        Generates a new question matching the target parameters along with the generation trace.
        """
        pass

class TargetSelector(ABC):
    """
    Abstract Base Class for selecting optimal target properties (TargetProfile)
    for the next question to synthesize to maximize the benchmark's information scale.
    """
    @abstractmethod
    def select_target(self, benchmark: Benchmark, response_matrix: pd.DataFrame) -> TargetProfile:
        """
        Analyzes the current calibration state and response data to select the optimal
        difficulty and load targets.
        """
        pass

from enum import Enum

class PresentationStyle(str, Enum):
    POSITIVE_UNBOUNDED = "POSITIVE_UNBOUNDED"
    SCALE_10 = "SCALE_10"

@dataclass
class NearbyExamplePrompterConfig:
    generator_model: str = "openai/gpt-4o"
    temperature: float = 1.0
    thinking_budget: int = 1024
    max_tokens: int = 4096
    p: float = 2.0
    num_examples: int = 3
    num_questions: int = 1
    min_discernability: Optional[float] = None
    max_discernability: Optional[float] = None
    detailed_analysis_prompt: bool = False
    seed: int = None
    type: str = "nearby_example_prompter"

@dataclass
class ScaledExamplePrompterConfig:
    generator_model: str = "openai/gpt-4o"
    temperature: float = 1.0
    thinking_budget: int = 1024
    max_tokens: int = 4096
    p: float = 2.0
    num_examples: int = 3
    num_questions: int = 1
    min_difficulty: float = -3.0
    min_discernability: Optional[float] = None
    max_discernability: Optional[float] = None
    presentation: PresentationStyle = PresentationStyle.SCALE_10
    seed: int = None
    double_ended: bool = False
    detailed_analysis_prompt: bool = False
    type: str = "scaled_example_prompter"

@dataclass
class IncreaseDifficultyPrompterConfig:
    generator_model: str = "openai/gpt-4o"
    temperature: float = 1.0
    thinking_budget: int = 0
    max_tokens: int = 4096
    p: float = 2.0
    delta_percent: float = 0.25
    num_questions: int = 1
    min_difficulty: float = -3.0
    min_discernability: Optional[float] = None
    max_discernability: Optional[float] = None
    seed: int = None
    type: str = "increase_difficulty_prompter"

@dataclass
class AddOptionPrompterConfig:
    generator_model: str = "openai/gpt-4o"
    temperature: float = 1.0
    thinking_budget: int = 0
    max_tokens: int = 4096
    p: float = 2.0
    num_questions: int = 1
    min_difficulty: float = -3.0
    min_discernability: Optional[float] = None
    max_discernability: Optional[float] = None
    selector_offset: float = 0.0
    seed: int = None
    type: str = "add_option_prompter"

# PrompterConfig Union definition
PrompterConfig = Union[NearbyExamplePrompterConfig, ScaledExamplePrompterConfig, IncreaseDifficultyPrompterConfig, AddOptionPrompterConfig]

@dataclass
class TrivialVerifierConfig:
    type: str = "trivial_verifier"

@dataclass
class LLMVerifierConfig:
    model: str
    temperature: float = 0.0
    thinking_budget: int = 1024
    max_tokens: int = 2048
    type: str = "llm_verifier"

# VerifierConfig Union definition
VerifierConfig = Union[TrivialVerifierConfig, LLMVerifierConfig]

@dataclass
class MidpointTargetSelectorConfig:
    model_a: str
    model_b: str
    type: str = "midpoint_target_selector"

# TargetSelectorConfig Union definition
TargetSelectorConfig = MidpointTargetSelectorConfig

@dataclass
class QuestionGeneratorConfig:
    """
    Configuration for the QuestionGenerator.
    """
    prompter: PrompterConfig
    verifier: VerifierConfig
    target_selector: TargetSelectorConfig
