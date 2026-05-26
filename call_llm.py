import traceback
from interfaces import LLMTrace


def call_llm(
    model: str, 
    system_prompt: str, 
    user_prompt: str, 
    temperature: float,
    response_format: str = "text",
    thinking_budget: int = 1,
    max_tokens: int = 4096,
    seed: int = None
) -> LLMTrace:
    if "/" not in model:
        raise ValueError(
            f"Model string must be in 'provider/model_name' format (e.g., 'openai/gpt-4o'), "
            f"got: '{model}'"
        )
    
    provider_raw, model_name = model.split("/", 1)
    provider = provider_raw.lower().strip()
    model_name = model_name.strip()

    if provider in ["openai", "gpt"]:
        import openai
        client = openai.OpenAI()
        
        kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        
        if thinking_budget > 0:
            kwargs["reasoning_effort"] = "medium"
        else:
            kwargs["temperature"] = temperature

        if max_tokens > 0:
            kwargs["max_completion_tokens"] = max_tokens

        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        if seed is not None:
            kwargs["seed"] = seed

        try:
            response = client.chat.completions.create(**kwargs)
        except openai.APIError as e:
            print(f"  ❌ [LLM] OpenAI API error: {type(e).__name__}: {e}")
            print(f"       model={model_name} prompt_len={len(user_prompt)}")
            traceback.print_exc()
            return LLMTrace(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                thinking=None,
                raw_output=""
            )
        except Exception as e:
            print(f"  ❌ [LLM] Unexpected error calling OpenAI: {type(e).__name__}: {e}")
            traceback.print_exc()
            return LLMTrace(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                thinking=None,
                raw_output=""
            )

        raw_output = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        usage = response.usage
        print(f"  [LLM] model={model_name} finish={finish_reason} "
              f"prompt_tokens={usage.prompt_tokens if usage else '?'} "
              f"completion_tokens={usage.completion_tokens if usage else '?'} "
              f"output_len={len(raw_output)}")

        if finish_reason and finish_reason != "stop":
            print(f"  ⚠️  [LLM] Non-standard finish_reason: {finish_reason}")

        # Extract OpenAI reasoning details if present in API usage details
        thinking = None
        if hasattr(response, "usage") and response.usage is not None:
            if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details is not None:
                details = response.usage.completion_tokens_details
                if hasattr(details, "reasoning_tokens") and details.reasoning_tokens is not None:
                    if details.reasoning_tokens > 0:
                        thinking = f"OpenAI reasoning tokens: {details.reasoning_tokens}"
                        
        return LLMTrace(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            thinking=thinking,
            raw_output=raw_output
        )
        
    elif provider in ["anthropic", "claude"]:
        import anthropic
        client = anthropic.Anthropic()
        
        kwargs = {
            "model": model_name,
            "max_tokens": max_tokens,  # Driven dynamically by config parameters
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        }
        
        if thinking_budget > 0:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            kwargs["temperature"] = 1.0
        else:
            kwargs["temperature"] = temperature

        try:
            response = client.messages.create(**kwargs)
        except anthropic.APIError as e:
            print(f"  ❌ [LLM] Anthropic API error: {type(e).__name__}: {e}")
            print(f"       model={model_name} prompt_len={len(user_prompt)}")
            traceback.print_exc()
            return LLMTrace(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                thinking=None,
                raw_output=""
            )
        except Exception as e:
            print(f"  ❌ [LLM] Unexpected error calling Anthropic: {type(e).__name__}: {e}")
            traceback.print_exc()
            return LLMTrace(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                thinking=None,
                raw_output=""
            )
        
        thinking_content = ""
        text_content = ""
        finish_reason = getattr(response, "stop_reason", None)
        usage = getattr(response, "usage", None)

        for block in response.content:
            if block.type == "thinking":
                thinking_content += block.thinking
            elif block.type == "text":
                text_content += block.text

        print(f"  [LLM] model={model_name} finish={finish_reason} "
              f"input_tokens={usage.input_tokens if usage else '?'} "
              f"output_tokens={usage.output_tokens if usage else '?'} "
              f"output_len={len(text_content)}")

        return LLMTrace(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            thinking=thinking_content if thinking_content else None,
            raw_output=text_content
        )
        
    else:
        raise ValueError(f"Unsupported LLM provider parsed: '{provider}'")

def extract_json_from_text(text: str) -> dict:
    """
    Gracefully extracts and parses a JSON dictionary block from raw model output.
    Handles both markdown code fences (```json ... ```) and raw string braces.
    Returns None if no valid JSON can be extracted.
    """
    import json
    import re

    if not text:
        print(f"  ⚠️  Empty model output, cannot extract JSON")
        return None

    # Try extracting contents between first '{' and last '}'
    try:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx + 1]
            return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON decode error (brace extraction): {e}")
    except Exception as e:
        print(f"  ⚠️  Unexpected error during JSON extraction: {type(e).__name__}: {e}")
        
    # Fallback to code fence extraction
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON decode error (code fence extraction): {e}")
        except Exception as e:
            print(f"  ⚠️  Unexpected error during code fence JSON extraction: {type(e).__name__}: {e}")
            
    print(f"  ⚠️  Could not parse JSON from model output (length={len(text)}, preview={text[:200]!r})")
    return None
