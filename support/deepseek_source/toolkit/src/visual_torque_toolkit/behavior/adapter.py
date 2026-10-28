"""Model adapters for batched multimodal candidate scoring."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import time
from typing import Any, Protocol, Sequence

from .data import BehaviorExample


@dataclass(frozen=True, slots=True)
class CandidateScoreResult:
    logprob_by_label: dict[str, float]
    scoring_mode: str
    latency_seconds: float


@dataclass(frozen=True, slots=True)
class GenerationResult:
    raw_response: str
    parsed_content: str


class ModelAdapter(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    def score_batch(
        self, examples: Sequence[BehaviorExample]
    ) -> list[CandidateScoreResult]: ...

    def generate_batch(
        self, examples: Sequence[BehaviorExample]
    ) -> list[GenerationResult]: ...


class HuggingFaceMultimodalAdapter:
    """A Transformers adapter shared by Qwen3.5 and Gemma 4.

    The legal completions are the neutral labels ``A``, ``B``, and ``C``. On
    both target tokenizers these are normally single tokens, allowing all three
    conditional likelihoods to be read from one forward pass. A slower exact
    multi-token fallback is retained for other tokenizers.
    """

    def __init__(
        self,
        model_id: str,
        *,
        revision: str = "main",
        dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except (ImportError, RuntimeError) as error:
            details = str(error)
            torchaudio_hint = ""
            if "torchaudio" in details.lower():
                torchaudio_hint = (
                    " VisualTorque is image-only: uninstall torchaudio, restart the "
                    "Python/Colab session, and retry."
                )
            raise RuntimeError(
                "Behavior inference requires torch, torchvision, transformers, "
                "and accelerate. Install the toolkit's 'behavior' extra."
                f"{torchaudio_hint} Original import error: {details}"
            ) from error

        self._torch = torch
        self.model_id = model_id
        self.revision = revision
        self.dtype_name = dtype
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = self.processor.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        torch_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"
        load_kwargs = {
            "revision": revision,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            "dtype": torch_dtype,
        }
        try:
            self.model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs)
        except TypeError:
            # Compatibility with Transformers releases that still use
            # ``torch_dtype`` instead of ``dtype``.
            load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
            self.model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs)
        self.model.eval()
        self._candidate_plan_cache: dict[
            tuple[str, ...], tuple[tuple[int, ...], dict[str, int]] | None
        ] = {}
        self._literal_completion_cache: dict[str, tuple[int, ...]] = {}

        resolved_revision = getattr(self.model.config, "_commit_hash", None)
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self._metadata = {
            "adapter": type(self).__name__,
            "model_id": model_id,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "dtype": dtype,
            "device_map": device_map,
            "parameter_count": parameter_count,
            "torch_version": torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def _input_device(self) -> Any:
        for parameter in self.model.parameters():
            if parameter.device.type != "meta":
                return parameter.device
        return self.model.device

    def _template(self, conversations: Any, *, add_generation_prompt: bool) -> Any:
        kwargs = {
            "add_generation_prompt": add_generation_prompt,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            # Transformers 5 requires kwargs forwarded to processor.__call__
            # to live under processor_kwargs rather than loose **kwargs.
            "processor_kwargs": {"padding": True},
            "enable_thinking": False,
        }
        try:
            return self.processor.apply_chat_template(conversations, **kwargs)
        except TypeError as error:
            # Some older chat templates do not expose thinking control. Refuse
            # this fallback for target families where hidden reasoning would
            # change the experiment, but permit it for genuinely non-thinking
            # models.
            model_name = self.model_id.lower()
            if "qwen3.5" in model_name or "gemma-4" in model_name:
                raise RuntimeError(
                    "The installed Transformers/chat template cannot disable "
                    "thinking for this target model. Upgrade Transformers."
                ) from error
            kwargs.pop("enable_thinking")
            return self.processor.apply_chat_template(conversations, **kwargs)

    def _to_device(self, inputs: Any) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(self._input_device)
        return {
            key: value.to(self._input_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _candidate_plan(
        self, example: BehaviorExample
    ) -> tuple[tuple[int, ...], dict[str, int]] | None:
        """Derive label tokens from the actual chat template, not raw encoding.

        Thinking-capable templates may insert a fixed response-channel prefix.
        That prefix is teacher-forced before reading the A/B/C distribution.
        """

        key = tuple(example.labels)
        if key in self._candidate_plan_cache:
            return self._candidate_plan_cache[key]

        prompt_inputs = self._template([example.messages()], add_generation_prompt=True)
        prompt_ids = prompt_inputs["input_ids"][0]
        prompt_mask = prompt_inputs["attention_mask"][0].bool()
        prompt_tokens = prompt_ids[prompt_mask].tolist()
        suffixes: dict[str, list[int]] = {}
        for label in example.labels:
            full_messages = example.messages() + [
                {"role": "assistant", "content": [{"type": "text", "text": label}]}
            ]
            full_inputs = self._template([full_messages], add_generation_prompt=False)
            full_ids = full_inputs["input_ids"][0]
            full_mask = full_inputs["attention_mask"][0].bool()
            full_tokens = full_ids[full_mask].tolist()
            if full_tokens[: len(prompt_tokens)] != prompt_tokens:
                # Gemma 4 deliberately renders a different prefix for an
                # empty model turn and a completed model turn. In particular,
                # disabling thinking puts an empty thought channel in the
                # generation prompt, while re-rendering a completed assistant
                # message omits that channel. The legal answer is nevertheless
                # generated immediately after the exact generation prompt, so
                # score its literal continuation instead of treating this
                # template difference as an error.
                plan = self._literal_candidate_plan(example.labels)
                self._candidate_plan_cache[key] = plan
                return plan
            suffixes[label] = full_tokens[len(prompt_tokens) :]

        values = list(suffixes.values())
        prefix_length = 0
        while all(
            prefix_length < len(value) and value[prefix_length] == values[0][prefix_length]
            for value in values
        ):
            prefix_length += 1
        suffix_length = 0
        while all(
            suffix_length < len(value) - prefix_length
            and value[-1 - suffix_length] == values[0][-1 - suffix_length]
            for value in values
        ):
            suffix_length += 1

        differing: dict[str, list[int]] = {}
        for label, value in suffixes.items():
            end = len(value) - suffix_length if suffix_length else len(value)
            differing[label] = value[prefix_length:end]
        if any(len(tokens) != 1 for tokens in differing.values()):
            self._candidate_plan_cache[key] = None
            return None
        plan = (
            tuple(values[0][:prefix_length]),
            {label: tokens[0] for label, tokens in differing.items()},
        )
        self._candidate_plan_cache[key] = plan
        return plan

    def _literal_completion_tokens(self, label: str) -> tuple[int, ...]:
        """Tokenize the text emitted after the exact generation prompt.

        This cache is initialized lazily as well as in ``__init__`` so a model
        already loaded in Colab can adopt a hot-reloaded adapter class without
        spending time loading its weights again.
        """

        cache = getattr(self, "_literal_completion_cache", None)
        if cache is None:
            cache = {}
            self._literal_completion_cache = cache
        if label in cache:
            return cache[label]

        encoded = self.processor.tokenizer(label, add_special_tokens=False)
        token_ids = encoded["input_ids"]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], (list, tuple)):
            if len(token_ids) != 1:
                raise RuntimeError(
                    f"tokenizer unexpectedly returned a batch for label {label!r}"
                )
            token_ids = token_ids[0]
        tokens = tuple(int(token_id) for token_id in token_ids)
        if not tokens:
            raise RuntimeError(f"legal label {label!r} tokenized to no tokens")
        cache[label] = tokens
        return tokens

    def _literal_candidate_plan(
        self, labels: Sequence[str]
    ) -> tuple[tuple[int, ...], dict[str, int]] | None:
        """Return a next-token plan when every literal label is one token."""

        tokens_by_label = {
            label: self._literal_completion_tokens(label) for label in labels
        }
        if any(len(tokens) != 1 for tokens in tokens_by_label.values()):
            return None
        return (), {
            label: tokens[0] for label, tokens in tokens_by_label.items()
        }

    def _append_text_prefix(self, inputs: Any, prefix: tuple[int, ...]) -> Any:
        if not prefix:
            return inputs
        torch = self._torch
        batch_size = inputs["input_ids"].shape[0]
        prefix_tensor = torch.tensor(
            prefix,
            dtype=inputs["input_ids"].dtype,
            device=inputs["input_ids"].device,
        ).unsqueeze(0).expand(batch_size, -1)
        inputs["input_ids"] = torch.cat((inputs["input_ids"], prefix_tensor), dim=1)
        attention_extension = torch.ones(
            (batch_size, len(prefix)),
            dtype=inputs["attention_mask"].dtype,
            device=inputs["attention_mask"].device,
        )
        inputs["attention_mask"] = torch.cat(
            (inputs["attention_mask"], attention_extension), dim=1
        )
        # Appended completion tokens are plain text. Keep every per-token type
        # tensor aligned with input_ids: Qwen3.5 uses mm_token_type_ids while
        # some other multimodal families expose token_type_ids.
        for token_type_key in ("token_type_ids", "mm_token_type_ids"):
            token_types = inputs.get(token_type_key)
            if token_types is not None and token_types.ndim == 2:
                text_types = torch.zeros(
                    (batch_size, len(prefix)),
                    dtype=token_types.dtype,
                    device=token_types.device,
                )
                inputs[token_type_key] = torch.cat(
                    (token_types, text_types), dim=1
                )
        # Let each architecture derive its own multimodal positions for the
        # newly appended fixed response prefix.
        inputs.pop("position_ids", None)
        return inputs

    def score_batch(
        self, examples: Sequence[BehaviorExample]
    ) -> list[CandidateScoreResult]:
        if not examples:
            return []
        if any(example.labels != examples[0].labels for example in examples):
            raise ValueError("all examples in one score batch must share legal labels")
        plan = self._candidate_plan(examples[0])
        if plan is None:
            return [self._score_multitoken(example) for example in examples]
        fixed_prefix, token_by_label = plan

        conversations = [example.messages() for example in examples]
        inputs = self._append_text_prefix(
            self._to_device(
                self._template(conversations, add_generation_prompt=True)
            ),
            fixed_prefix,
        )
        started = time.perf_counter()
        with self._torch.inference_mode():
            output = self.model(**inputs, use_cache=False)
            next_token_logprobs = self._torch.log_softmax(
                output.logits[:, -1, :].float(), dim=-1
            )
        elapsed = time.perf_counter() - started
        per_example_latency = elapsed / len(examples)

        results = []
        for row, example in enumerate(examples):
            scores = {
                label: float(next_token_logprobs[row, token_by_label[label]].item())
                for label in example.labels
            }
            results.append(
                CandidateScoreResult(
                    scores, "single_token_next_token", per_example_latency
                )
            )
        return results

    def _score_multitoken(self, example: BehaviorExample) -> CandidateScoreResult:
        """Teacher-force a literal label after the exact generation prompt.

        A completed assistant turn is intentionally not re-rendered here:
        templates such as Gemma 4's need not preserve an empty generation turn
        as a prefix of completed history.
        """

        prompt_inputs = self._template(
            [example.messages()], add_generation_prompt=True
        )
        prompt_width = prompt_inputs["input_ids"].shape[1]
        scores: dict[str, float] = {}
        started = time.perf_counter()

        for label in example.labels:
            completion_tokens = self._literal_completion_tokens(label)
            # Work on a shallow copy because _append_text_prefix replaces text
            # tensors in place. Image tensors are read-only and may be shared.
            full_inputs = self._append_text_prefix(
                dict(prompt_inputs), completion_tokens
            )
            device_inputs = self._to_device(full_inputs)
            with self._torch.inference_mode():
                logits = self.model(**device_inputs, use_cache=False).logits[0].float()
                logprobs = self._torch.log_softmax(logits, dim=-1)
            token_score = 0.0
            for offset, token_id in enumerate(completion_tokens):
                token_score += float(
                    logprobs[prompt_width + offset - 1, token_id].item()
                )
            scores[label] = token_score

        return CandidateScoreResult(
            scores,
            "literal_multitoken_teacher_forced",
            time.perf_counter() - started,
        )

    @staticmethod
    def _content_text(parsed: Any) -> str | None:
        if isinstance(parsed, str):
            return parsed.strip()
        if isinstance(parsed, dict):
            content = parsed.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                texts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                return "".join(texts).strip()
        return None

    def _parse_generated_content(
        self,
        raw: str,
        prompt_tokens: Any,
        generated_tokens: Any,
    ) -> str:
        """Extract visible assistant text without assuming a response template.

        In Transformers 5, ``parse_response`` can exist on the tokenizer even
        when the loaded model provides neither ``response_template`` nor the
        legacy ``response_schema``. Calling it in that state raises an
        ``AttributeError``. Follow the official pipelines and invoke it only
        when the tokenizer actually declares a response parser configuration.
        """

        tokenizer = self.processor.tokenizer
        response_configuration = getattr(tokenizer, "response_template", None)
        if response_configuration is None:
            response_configuration = getattr(tokenizer, "response_schema", None)
        parser = (
            getattr(tokenizer, "parse_response", None)
            if response_configuration is not None
            else None
        )
        if parser is not None:
            prefix_text = self.processor.decode(
                prompt_tokens,
                skip_special_tokens=False,
            )
            try:
                parsed_content = self._content_text(
                    parser(raw, prefix=prefix_text)
                )
            except TypeError:
                try:
                    parsed_content = self._content_text(parser(raw))
                except (AttributeError, TypeError, ValueError):
                    parsed_content = None
            except (AttributeError, ValueError):
                parsed_content = None
            if parsed_content is not None:
                return parsed_content

        return self.processor.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

    def generate_batch(self, examples: Sequence[BehaviorExample]) -> list[GenerationResult]:
        if not examples:
            return []
        inputs = self._to_device(
            self._template(
                [example.messages() for example in examples],
                add_generation_prompt=True,
            )
        )
        input_width = inputs["input_ids"].shape[1]
        with self._torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=16,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        results = []
        for index, row in enumerate(outputs):
            generated_tokens = row[input_width:]
            raw = self.processor.decode(
                generated_tokens, skip_special_tokens=False
            ).strip()
            prompt_tokens = inputs["input_ids"][index][
                inputs["attention_mask"][index].bool()
            ]
            parsed_content = self._parse_generated_content(
                raw,
                prompt_tokens,
                generated_tokens,
            )
            results.append(GenerationResult(raw, parsed_content))
        return results
