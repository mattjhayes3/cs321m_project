import json
import numpy as np
from typing import List, Any, Optional, Dict
from interfaces import (
    Question, TargetProfile, Benchmark, Prompter,
    NearbyExamplePrompterConfig, ScaledExamplePrompterConfig,
    IncreaseDifficultyPrompterConfig, AddOptionPrompterConfig,
    PrompterResponse, LLMTrace
)
from call_llm import call_llm, extract_json_from_text
from utils import format_question


# ────────────────────────────────────────────────────────────────
# SHARED UTILITIES / CONSTRAINTS AND LOGIC
# ────────────────────────────────────────────────────────────────

JSON_RESPONSE_FORMAT_INSTRUCTION = """Respond STRICTLY with a single JSON object wrapping a list of questions under the 'questions' key; use exclusively basic ASCII characters, e.g.:
{
  "questions": [
    {
      "question_text": "<question text string>",
      "options": ["choice A", "choice B", "choice C", "choice D"],
      "correct_answer": "A"
    }
  ]
}"""

SIMPLE_EXEMPLAR_INSTRUCTION = (
    "Observe the relationship between vocabulary complexity, reasoning steps, "
    "and distractors in the exemplars.\n\n"
)

DETAILED_EXEMPLAR_INSTRUCTION = (
    "Carefully analyze each example question to understand its difficulty, "
    "thinking about the following questions before generating your own:\n"
    "* How many reasoning steps are needed to solve the question?\n"
    "* What definitions or concepts are needed? At what grade level are they"
    "typically taught? How memorable are they? \n"
    "* How many distractors are frivolous, and how many are plausible? How plausible?\n"
    "* Can the question be solved with pure word association or would that "
    "lead to one of the distractors?\n\n"
)


def compute_lp_distance(
    q: Question,
    target_diff: float,
    p: float,
    target_factor_loadings: Optional[List[Any]] = None
) -> float:
    """
    Compute Minkowski (L_p) distance between a question's attributes and targets.
    Optionally accounts for multidimensional factor loadings.
    """
    dist_sum = abs(q.difficulty - target_diff) ** p

    # Compute factor loadings loading distance if present
    if target_factor_loadings and q.factor_loadings:
        t_factors = {f.name: f.loading for f in target_factor_loadings}
        q_factors = {f.name: f.loading for f in q.factor_loadings}
        all_names = set(t_factors.keys()) | set(q_factors.keys())

        for name in all_names:
            t_load = t_factors.get(name, 0.0)
            q_load = q_factors.get(name, 0.0)
            dist_sum += abs(q_load - t_load) ** p

    return dist_sum ** (1.0 / p)


def sample_by_lp_distance(
    candidates: List[Question],
    target_diff: float,
    p: float,
    num_to_sample: int = 1,
    target_factor_loadings: Optional[List[Any]] = None,
    disallow_indices: Optional[List[int]] = None
) -> List[int]:
    """
    Compute Minkowski (L_p) distances for all candidates from target_diff,
    restrict the candidates to the Top-K (Top-20) closest to prevent density bias,
    and sample `num_to_sample` indices proportional to inverse distance.
    """
    distances = [
        compute_lp_distance(q, target_diff, p, target_factor_loadings)
        for q in candidates
    ]
    distances = np.array(distances)

    epsilon = 1e-6
    weights = 1.0 / (distances + epsilon)

    if disallow_indices:
        for idx in disallow_indices:
            weights[idx] = 0.0

    # Greedy Top-K filter: only allow sampling from the Top-20 closest active candidates
    # (or less if candidate pool is smaller)
    active_indices = np.nonzero(weights > 0)[0]
    if len(active_indices) == 0:
        raise ValueError("No active candidates available for sampling!")
        
    # Find the indices of the Top-20 closest questions among active ones
    top_k = min(20, len(active_indices))
    # Sort active indices by their distances ascending
    sorted_active = sorted(active_indices, key=lambda idx: distances[idx])
    allowed_set = set(sorted_active[:top_k])

    # Nullify weights of anything outside the top-K allowed set
    for idx in range(len(weights)):
        if idx not in allowed_set:
            weights[idx] = 0.0

    probs = weights / np.sum(weights)

    # Sample without replacement
    selected = np.random.choice(
        len(candidates),
        size=num_to_sample,
        replace=False,
        p=probs
    )
    return list(selected)


def filter_by_discernability(
    candidates: List[Question],
    min_discernability: Optional[float],
    max_discernability: Optional[float]
) -> List[Question]:
    """
    Filters candidate questions based on minimum and maximum estimated discernability (discrimination) parameters.
    """
    if min_discernability is None and max_discernability is None:
        return candidates

    filtered = []
    for q in candidates:
        if q.discrimination is not None:
            if min_discernability is not None and q.discrimination < min_discernability:
                continue
            if max_discernability is not None and q.discrimination > max_discernability:
                continue
        filtered.append(q)
    return filtered


def format_exemplars(
    selected: List[Question],
    scaled_difficulties: Optional[Dict[str, Any]] = None
) -> str:
    """
    Formats selected exemplar questions cleanly for the LLM prompt.
    Optionally includes scaled difficulties if provided.
    """
    blocks = []
    for i, q in enumerate(selected):
        header = f"Example {i+1}:"
        if scaled_difficulties and q.id in scaled_difficulties:
            header = f"Example {i+1} (Difficulty Rating: {scaled_difficulties[q.id]}):"
        blocks.append(f"{header}\n{format_question(q, include_answer=True)}")
    return "\n\n----------------------------------------\n\n".join(blocks)


def parse_generated_questions(
    raw_output: str,
    target_difficulty: float,
    target_factor_loadings: Optional[List[Any]] = None
) -> List[Question]:
    """
    Gracefully parse a JSON raw output string containing generated questions.
    Enforces strict field presence, maps synonym keys, and returns a list of verified Question objects.
    """
    raw_data = extract_json_from_text(raw_output)
    if raw_data is None:
        print("  ⚠️  No valid JSON extracted from LLM output")
        return []

    # Normalize raw_data to a list of question dicts
    if isinstance(raw_data, dict):
        # 1. Explicit wrapper key check
        if "questions" in raw_data and isinstance(raw_data["questions"], list):
            raw_data = raw_data["questions"]
        else:
            # 2. Robust heuristic: find a key that is a list of DICTIONARIES
            for val in raw_data.values():
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                    raw_data = val
                    break
            else:
                # 3. Fallback: if the dict itself is a single question
                raw_data = [raw_data]

    questions = []
    for idx, data in enumerate(raw_data):
        # Handle case where data is a string (try to parse as JSON)
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                print(f"  ⚠️  Skipping non-dict question {idx}: {type(data).__name__}")
                continue
        if not isinstance(data, dict):
            print(f"  ⚠️  Skipping non-dict question {idx}: {type(data).__name__}")
            continue

        # Robust key lookup for synonym keys, but enforce strict presence (no silent defaults)
        q_text = (data.get("question_text")
                  or data.get("question")
                  or data.get("text")
                  or data.get("stem"))
        opts = (data.get("options")
                or data.get("choices")
                or data.get("answers"))
        ans = (data.get("correct_answer")
               or data.get("answer")
               or data.get("correct"))

        if not q_text or not opts or ans is None:
            print(f"  ⚠️  Skipping malformed question {idx}: missing required fields (question_text, options, or correct_answer)")
            continue

        candidate = Question(
            id=f"gen_{idx}_{np.random.randint(1000000)}",
            question_text=q_text,
            options=opts,
            correct_answer=str(ans).strip().upper(),
            difficulty=target_difficulty,
            factor_loadings=target_factor_loadings,
            calibrated=False
        )
        questions.append(candidate)

    return questions


# ────────────────────────────────────────────────────────────────
# CONCRETE PROMPTER IMPLEMENTATIONS
# ────────────────────────────────────────────────────────────────

class NearbyExamplePrompter(Prompter):
    """
    Combined Selector and Prompter that retrieves seed exemplars proportional to their L* distance,
    formats them cleanly, issues the LLM call to generate a batch of K target questions, and parses the JSON list returns.
    Fully driven by NearbyExamplePrompterConfig.
    """
    def __init__(self, config: NearbyExamplePrompterConfig):
        self.config = config

    def _select_examples(self, benchmark: Benchmark, target_profile: TargetProfile, exclude_ids: Optional[List[str]] = None) -> List[Question]:
        # Filter by discernability (discrimination) thresholds if set
        candidates = filter_by_discernability(benchmark.questions, self.config.min_discernability, self.config.max_discernability)
        if exclude_ids:
            candidates = [c for c in candidates if c.id not in exclude_ids]

        print(f"  [Prompter] NearbyExamplePrompter filtered candidates: {len(benchmark.questions)} -> {len(candidates)}")
        assert len(candidates) >= self.config.num_examples

        target_diff = target_profile.target_difficulty
        selected_indices = sample_by_lp_distance(
            candidates=candidates,
            target_diff=target_diff,
            p=self.config.p,
            num_to_sample=self.config.num_examples,
            target_factor_loadings=target_profile.target_factor_loadings
        )

        return [candidates[idx] for idx in selected_indices]

    def get_examples(self, benchmark: Benchmark, target_profile: TargetProfile, exclude_ids: Optional[List[str]] = None) -> PrompterResponse:
        # 1. Select optimal exemplars
        selected = self._select_examples(benchmark, target_profile, exclude_ids=exclude_ids)
        assert len(selected) == self.config.num_examples

        target_diff = target_profile.target_difficulty

        # 2. Format Exemplars using shared helper
        exemplars_str = format_exemplars(selected)
        exemplar_instruction = (
            DETAILED_EXEMPLAR_INSTRUCTION if self.config.detailed_analysis_prompt
            else SIMPLE_EXEMPLAR_INSTRUCTION
        )

        system_prompt = (
            "You are a world-class test designer and science curriculum developer.\n"
            "Your task is to write multiple-choice science questions for a benchmark.\n"
            "Each question must have exactly four options (A, B, C, D) and exactly one correct answer.\n"
            "You will be given several seed questions (exemplars).\n"
            f"{exemplar_instruction}"
            f"{JSON_RESPONSE_FORMAT_INSTRUCTION}"
        )

        user_prompt = (
            f"Here are the seed exemplars representing the benchmark:\n\n"
            f"{exemplars_str}\n\n"
            f"Write exactly {self.config.num_questions} unique new multiple-choice science question(s) "
            "corresponding to the target difficulty demonstrated in the exemplars.\n"
            "Ensure that the question text is direct and clear, but requires appropriate reasoning/knowledge."
        )

        # 3. LLM Generation
        trace = call_llm(
            model=self.config.generator_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            response_format="json",
            thinking_budget=self.config.thinking_budget,
            max_tokens=self.config.max_tokens,
            seed=self.config.seed
        )

        # Refactored to use shared parsing logic
        questions = parse_generated_questions(trace.raw_output, target_diff, target_profile.target_factor_loadings)
        return PrompterResponse(questions=questions, trace=trace, exemplars=selected)


class ScaledExamplePrompter(Prompter):
    """
    Exemplar Selector and Prompter that retrieves seed exemplars uniformly distributed
    from min_difficulty to target_difficulty, scales their difficulties according to
    a presentation style (POSITIVE_UNBOUNDED or SCALE_10), and prompts the generator LLM.
    """
    def __init__(self, config: ScaledExamplePrompterConfig):
        self.config = config
        assert self.config.num_examples >= 2, "num_examples must be at least 2"

    def _select_examples(self, benchmark: Benchmark, target_profile: TargetProfile, exclude_ids: Optional[List[str]] = None) -> List[Question]:
        # Filter by discernability (discrimination) thresholds if set
        candidates = filter_by_discernability(benchmark.questions, self.config.min_discernability, self.config.max_discernability)
        if exclude_ids:
            candidates = [c for c in candidates if c.id not in exclude_ids]

        print(f"  [Prompter] ScaledExamplePrompter filtered candidates: {len(benchmark.questions)} -> {len(candidates)}")
        assert len(candidates) >= self.config.num_examples

        target_diff = target_profile.target_difficulty
        min_diff = self.config.min_difficulty
        
        # Enforce min_diff is strictly less than target_diff
        if min_diff >= target_diff:
            min_diff = target_diff - 1.5

        # 1. Generate N selection points spaced across the distance to target_diff
        delta = target_diff - min_diff
        num_between = self.config.num_examples
        if self.config.double_ended:
            num_between -= 1

        points = [min_diff + (i / num_between) * delta for i in range(num_between)]
        if self.config.double_ended:
            points.append(target_diff)
            
        selected_indices = []

        for pj in points:
            sel = sample_by_lp_distance(
                candidates=candidates,
                target_diff=pj,
                p=self.config.p,
                num_to_sample=1,
                target_factor_loadings=target_profile.target_factor_loadings,
                disallow_indices=selected_indices
            )
            selected_indices.extend(sel)

        return [candidates[i] for i in selected_indices]

    def get_examples(self, benchmark: Benchmark, target_profile: TargetProfile, exclude_ids: Optional[List[str]] = None) -> PrompterResponse:
        from interfaces import PresentationStyle

        # 1. Select exemplars
        selected_qs = self._select_examples(benchmark, target_profile, exclude_ids=exclude_ids)
        target_diff = target_profile.target_difficulty

        # Find lowest difficulty among selected
        b_min = min(q.difficulty for q in selected_qs)
        
        # 2. Scale difficulties based on presentation style
        scaled_difficulties = {}
        scaled_target = 0.0

        if self.config.presentation == PresentationStyle.POSITIVE_UNBOUNDED:
            # Shift lowest selected to 0.0, round to nearest tenth
            for q in selected_qs:
                scaled_difficulties[q.id] = round(q.difficulty - b_min, 1)
            scaled_target = round(target_diff - b_min, 1)
            style_instruction = (
                "The difficulty rating starts at 0.0 (easiest example).\n"
                f"Your goal is to write new questions with difficulty rating of {scaled_target}."
            )
        elif self.config.presentation == PresentationStyle.SCALE_10:
            # Map b_min -> 1, target_diff -> 9
            # Solve: m * b_min + c = 1,  m * target_diff + c = 9
            denom = target_diff - b_min
            if abs(denom) < 1e-6:
                m = 1.0
                c = 1.0 - b_min
            else:
                m = 8.0 / denom
                c = 1.0 - m * b_min

            for q in selected_qs:
                val = m * q.difficulty + c
                scaled_difficulties[q.id] = int(round(val))
            scaled_target = 9
            style_instruction = (
                "The difficulty rating is a scale from 1 (extremely easy) to 10 (extremely hard).\n"
                f"Your goal is to write new questions at difficulty rating of {scaled_target}."
            )

        # 3. Format Exemplars using shared helper
        exemplars_str = format_exemplars(selected_qs, scaled_difficulties)

        if self.config.detailed_analysis_prompt:
            exemplar_instruction = DETAILED_EXEMPLAR_INSTRUCTION
        else:
            exemplar_instruction = SIMPLE_EXEMPLAR_INSTRUCTION

        system_prompt = (
            "You are an expert psychometrician and science curriculum developer.\n"
            "Your task is to write multiple-choice science questions for a benchmark.\n"
            "Each question must have exactly four options (A, B, C, D) and exactly one correct answer.\n"
            "You will be given several seed questions (exemplars) along with their difficulty ratings.\n"
            f"{exemplar_instruction}"
            f"{style_instruction}\n"
            f"{JSON_RESPONSE_FORMAT_INSTRUCTION}"
        )

        user_prompt = (
            f"Here are the seed exemplars representing the benchmark scale:\n\n"
            f"{exemplars_str}\n\n"
            f"Using the scale demonstrated above, write exactly {self.config.num_questions} unique new multiple-choice science "
            f"questions of similar topics but with a target difficulty rating of {scaled_target}.\n"
            "Ensure that the question text is direct and clear, but requires appropriate reasoning/knowledge "
            "corresponding to the target difficulty rating."
        )

        # 4. LLM Generation
        trace = call_llm(
            model=self.config.generator_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            response_format="json",
            thinking_budget=self.config.thinking_budget,
            max_tokens=self.config.max_tokens,
            seed=self.config.seed
        )

        # Refactored to use shared parsing logic
        questions = parse_generated_questions(trace.raw_output, target_diff, target_profile.target_factor_loadings)
        return PrompterResponse(questions=questions, trace=trace, exemplars=selected_qs)


class IncreaseDifficultyPrompter(Prompter):
    """
    Prompter that selects a single base exemplar at (target_difficulty - delta_percent * (target_difficulty - min_difficulty)),
    presents it, and instructs the LLM to rewrite it to be exactly delta_percent harder.
    """
    def __init__(self, config: IncreaseDifficultyPrompterConfig):
        self.config = config

    def _select_examples(self, benchmark: Benchmark, target_profile: TargetProfile, exclude_ids: Optional[List[str]] = None) -> List[Question]:
        # Filter by discernability thresholds
        candidates = filter_by_discernability(
            benchmark.questions, self.config.min_discernability, self.config.max_discernability
        )
        if exclude_ids:
            candidates = [c for c in candidates if c.id not in exclude_ids]

        print(f"  [Prompter] IncreaseDifficultyPrompter filtered candidates: {len(benchmark.questions)} -> {len(candidates)}")
        
        target_diff = target_profile.target_difficulty
        assert target_diff > self.config.min_difficulty, "Target difficulty must be strictly greater than minimum difficulty"
        
        # Calculate base difficulty to select near: target_diff - delta_percent * (target_diff - min_difficulty)
        target_base_diff = target_diff - self.config.delta_percent * (target_diff - self.config.min_difficulty)
        
        # Ensure base difficulty makes physical sense
        if target_base_diff >= target_diff:
            target_base_diff = target_diff - 0.5

        selected_indices = sample_by_lp_distance(
            candidates=candidates,
            target_diff=target_base_diff,
            p=self.config.p,
            num_to_sample=1,
            target_factor_loadings=target_profile.target_factor_loadings
        )
        return [candidates[selected_indices[0]]]

    def get_examples(self, benchmark: Benchmark, target_profile: TargetProfile, exclude_ids: Optional[List[str]] = None) -> PrompterResponse:
        # 1. Select base question
        selected_qs = self._select_examples(benchmark, target_profile, exclude_ids=exclude_ids)
        base_q = selected_qs[0]
        target_diff = target_profile.target_difficulty

        system_prompt = (
            "You are an expert psychometrician, science test designer, and curriculum developer.\n"
            "Your task is to read a given science question and rewrite it to be "
            f"{self.config.delta_percent:.0%} harder (increasing its difficulty rating).\n"
            "To increase the difficulty, you should:\n"
            "1. Introduce a deeper, more advanced scientific concept or a multi-step causal relationship.\n"
            "2. Increase the vocabulary precision and sentence complexity.\n"
            "3. Make the incorrect options (distractors) subtler, requiring closer discrimination to eliminate.\n\n"
            "The rewritten question must maintain the same overall topic and correct option, have exactly "
            "four choices (A, B, C, D) and exactly one correct answer.\n"
            f"{JSON_RESPONSE_FORMAT_INSTRUCTION}"
        )

        user_prompt = (
            "Here is the base science question:\n\n"
            f"{format_question(base_q, include_answer=True)}\n\n"
            f"Rewrite this question to make it about {self.config.delta_percent:.0%} harder.\n"
            f"Generate exactly {self.config.num_questions} rewritten version(s) in the requested JSON format."
        )

        trace = call_llm(
            model=self.config.generator_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            response_format="json",
            thinking_budget=self.config.thinking_budget,
            max_tokens=self.config.max_tokens,
            seed=self.config.seed
        )

        questions = parse_generated_questions(trace.raw_output, target_diff, target_profile.target_factor_loadings)
        return PrompterResponse(questions=questions, trace=trace, exemplars=[base_q])


class AddOptionPrompter(Prompter):
    """
    Prompter that selects a base question slightly below the target difficulty,
    and instructs the LLM to rewrite the question to add exactly one highly plausible
    new distractor option (expanding options from N to N+1).
    """
    def __init__(self, config: AddOptionPrompterConfig):
        self.config = config

    def _select_examples(self, benchmark: Benchmark, target_profile: TargetProfile, exclude_ids: Optional[List[str]] = None) -> List[Question]:
        # Filter by discernability thresholds
        candidates = filter_by_discernability(
            benchmark.questions, self.config.min_discernability, self.config.max_discernability
        )
        if exclude_ids:
            candidates = [c for c in candidates if c.id not in exclude_ids]

        print(f"  [Prompter] AddOptionPrompter filtered candidates: {len(benchmark.questions)} -> {len(candidates)}")
        
        target_diff = target_profile.target_difficulty
        assert target_diff > self.config.min_difficulty, "Target difficulty must be strictly greater than minimum difficulty"
        
        # Target base difficulty: assume N=4 options as default, so we want to select around target - 20% of scale + selector_offset
        target_base_diff = target_diff - 0.20 * (target_diff - self.config.min_difficulty) + self.config.selector_offset
        
        if target_base_diff >= target_diff:
            target_base_diff = target_diff - 0.5

        selected_indices = sample_by_lp_distance(
            candidates=candidates,
            target_diff=target_base_diff,
            p=self.config.p,
            num_to_sample=1,
            target_factor_loadings=target_profile.target_factor_loadings
        )
        return [candidates[selected_indices[0]]]

    def get_examples(self, benchmark: Benchmark, target_profile: TargetProfile, exclude_ids: Optional[List[str]] = None) -> PrompterResponse:
        # 1. Select base question
        selected_qs = self._select_examples(benchmark, target_profile, exclude_ids=exclude_ids)
        base_q = selected_qs[0]
        target_diff = target_profile.target_difficulty
        
        n_options = len(base_q.options)
        target_options = n_options + 1
        valid_letters = [chr(ord('A') + i) for i in range(target_options)]

        system_prompt = (
            "You are an expert psychometrician and science curriculum developer.\n"
            f"Your task is to read a given science question with exactly {n_options} choices, "
            f"and increase its psychometric difficulty by adding exactly ONE highly plausible, incorrect distractor choice to create a refined version with exactly {target_options} choices ({', '.join(valid_letters)}).\n\n"
            "Strict constraints for this psychometric task:\n"
            f"1. Do NOT modify the original question_text stem under any circumstances: '{base_q.question_text}'\n"
            f"2. Retain the original {n_options} choices exactly word-for-word: {base_q.options}\n"
            "3. Insert exactly one new incorrect distractor option into the 'options' list (it can be placed at any position, it does not need to be last).\n"
            "4. The new distractor must target a common misconception, be scientifically related to the topic, but remain incorrect.\n"
            "5. Update the 'correct_answer' to be the single uppercase letter (e.g. 'A', 'B', 'C', 'D', or 'E') matching the correct choice in the new options list.\n\n"
            f"{JSON_RESPONSE_FORMAT_INSTRUCTION}"
        )

        user_prompt = (
            "Here is the base question to refine:\n\n"
            "{\n"
            f"  \"question_text\": \"{base_q.question_text}\",\n"
            f"  \"options\": {json.dumps(base_q.options)},\n"
            f"  \"correct_answer\": \"{base_q.correct_answer}\"\n"
            "}\n\n"
            f"Generate a refined version of this question containing exactly {target_options} choices by adding exactly one incorrect option. Keep the question_text and original options word-for-word identical."
        )

        trace = call_llm(
            model=self.config.generator_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            response_format="json",
            thinking_budget=self.config.thinking_budget,
            max_tokens=self.config.max_tokens,
            seed=self.config.seed
        )

        questions = parse_generated_questions(trace.raw_output, target_diff, target_profile.target_factor_loadings)
        return PrompterResponse(questions=questions, trace=trace, exemplars=[base_q])
