# DSPy Codebase Overview

## What is DSPy?

**DSPy** (Declarative Self-improving Python) is a Stanford-NLP framework for **programming—rather than prompting—language models**. Instead of writing brittle prompt strings, you write structured Python programs. DSPy can then automatically **optimize** those programs (both prompts and model weights) to produce high-quality outputs.

> **Core idea:** You declare *what* you want (via `Signature`), compose modules (via `Module`/`Predict`), and let DSPy's optimizers (teleprompters) figure out *how* to best elicit that from an LM.

---

## Top-Level Repo Structure

| Directory/File | Purpose |
|---|---|
| `dspy/` | The core library package |
| `tests/` | Unit and integration tests |
| `docs/` | Documentation source (dspy.ai) |
| `.github/` | CI/CD workflows and issue templates |
| `pyproject.toml` | Package metadata and dependencies |
| `uv.lock` | Dependency lock file (via `uv`) |

---

## `dspy/` — Subdirectories at a Glance

| Subdirectory | Role |
|---|---|
| `signatures/` | Defines the typed input/output contract (`Signature`) describing what an LM call should do |
| `predict/` | Built-in LM-calling modules (`Predict`, `ChainOfThought`, `ReAct`, etc.) that execute against a signature |
| `primitives/` | Core building blocks: `Module`, `Example`, `Prediction`, and code interpreters |
| `adapters/` | Translates a `Signature` + inputs into actual LM prompt formats (Chat, JSON, XML, BAML, two-step) |
| `clients/` | Wraps LM provider APIs (OpenAI, Databricks, local models) and handles caching, embeddings, fine-tuning |
| `teleprompt/` | Optimizers ("teleprompters") that automatically tune prompts and/or weights (MIPRO, BootstrapFewShot, GRPO, SIMBA, etc.) |
| `evaluate/` | Tools to run evaluations of DSPy programs against datasets with scoring metrics |
| `retrievers/` | Retrieval modules for RAG pipelines (Databricks, Weaviate, embedding-based, etc.) |
| `datasets/` | Built-in benchmark datasets (HotpotQA, GSM8K, MATH, ALFWorld, etc.) |
| `utils/` | Shared utilities: callbacks, async/sync bridges, parallelism, caching, logging, MCP tool wrappers |
| `streaming/` | Infrastructure for streaming LM responses token-by-token to listeners |
| `propose/` | Generates candidate instruction proposals used by teleprompters during optimization |
| `dsp/` | Legacy lower-level DSP primitives and the global `settings` singleton |
| `experimental/` | Staging area for experimental/unstable features |

---

## How It All Fits Together

| Phase | Submodules Involved |
|---|---|
| 👤 **You (the Developer)** | Define a `Signature` → Build a `Module` → Configure an `LM` |
| ⚙️ **Inference** | Pick a predictor → Fetch RAG context → Format prompt → Call LM → Parse output → Execute code / stream tokens |
| 🔁 **Optimization Loop** | Load dataset → Evaluate → Propose instructions → Run optimizer → Fine-tune weights (feeds back into the module and LM) |
| 🛠️ **Shared Infrastructure** | Callbacks, async/sync bridges, save/load, and external tool support — cross-cutting all phases |

---

## File-by-File Reference

### `signatures/`

#### `signature.py`
Core typed `input → output` contract for any LM call, built on Pydantic.
- `class Signature` — base class; subclass this to define your task
- `class SignatureMeta` — metaclass managing field ordering, instructions, and type inference
  - `.input_fields` / `.output_fields` / `.fields` — field access properties
  - `.instructions` — the task description written in `__doc__`
- `Signature.with_instructions(instructions)` — return a new Signature with different instructions
- `Signature.with_updated_fields(name, type_, **kwargs)` — return a new Signature with updated field metadata
- `Signature.append / prepend / insert / delete(...)` — immutable field manipulation
- `Signature.dump_state() / load_state(state)` — serialization for save/load
- `make_signature(signature, instructions)` — programmatic factory (used internally)
- `ensure_signature(signature)` — coerce string or class to a Signature type

#### `field.py`
Defines the `InputField` and `OutputField` constructors that annotate Signature fields.
- `InputField(**kwargs)` — creates a Pydantic field tagged as an input
- `OutputField(**kwargs)` — creates a Pydantic field tagged as an output
- `move_kwargs(**kwargs)` — splits kwargs between Pydantic and DSPy-specific metadata
- `_translate_pydantic_field_constraints(**kwargs)` — renders Pydantic constraints as human-readable hints

#### `utils.py`
- `get_annotation_name(annotation)` — converts a type annotation to a readable string

---

### `predict/`

#### `predict.py`
The base module that calls an LM via an adapter against a signature — the workhorse of DSPy.
- `class Predict(Module, Parameter)` — the fundamental LM-calling unit
  - `.__init__(signature, callbacks, **config)` — attach signature and LM config
  - `.forward(**kwargs)` — synchronous call: preprocess → adapter → LM → postprocess
  - `.aforward(**kwargs)` — async equivalent
  - `._forward_preprocess(**kwargs)` — resolves LM, demos, config, and defaults
  - `._forward_postprocess(completions, signature)` — wraps completions into a `Prediction`, appends to trace
  - `.dump_state() / load_state(state)` — serialize/deserialize demos, signature, and LM config
  - `.reset()` — clear demos and traces
  - `.update_config(**kwargs)` — update per-call LM kwargs

#### `chain_of_thought.py`
Extends `Predict` by injecting a `reasoning` field before the answer.
- `class ChainOfThought(Module)`
  - `.__init__(signature, rationale_field, rationale_field_type)` — prepends a reasoning output field
  - `.forward(**kwargs)` / `.aforward(**kwargs)` — delegates to inner `Predict`

#### `react.py`
Implements the Reason+Act loop for tool-using agents.
- `class ReAct(Module)`
  - `.__init__(signature, tools, max_iters)` — wraps tools as `Tool` types
  - `.forward(**input_args)` / `.aforward(**input_args)` — iterative think/act/observe loop
  - `.truncate_trajectory(trajectory)` — override to handle context window overflows
  - `._format_trajectory(trajectory)` — renders the running thought/action/observation history

#### `program_of_thought.py`
Generates and executes Python code to answer questions.
- `class ProgramOfThought(Module)`
  - `.forward(**kwargs)` — generates code, executes it, retries on error

#### `code_act.py`
Agent that writes and runs code iteratively until a task is complete.
- `class CodeAct(Module)`
  - `.forward(**kwargs)` — code generation + execution loop with tool feedback

#### `refine.py`
Iteratively refines an output by critiquing and regenerating.
- `class Refine(Module)`
  - `.forward(**kwargs)` — runs the inner module, scores, retries up to N times

#### `best_of_n.py`
Samples N completions and picks the best via a reward function.
- `class BestOfN(Module)`
  - `.forward(**kwargs)` — runs N forward passes, applies reward, returns top result

#### `multi_chain_comparison.py`
Runs multiple reasoning chains and aggregates via majority vote / LM comparison.
- `class MultiChainComparison(Module)`
  - `.forward(**kwargs)` — aggregates multiple chain-of-thought outputs

#### `retry.py`
Wraps a predictor and retries on assertion failures.
- `class Retry(Module)`
  - `.forward(**kwargs)` — catches `DSPyAssertionError` and retries with feedback

#### `aggregation.py`
- `majority(predictions, field)` — returns the most common value across a list of predictions

#### `knn.py`
Retrieves k-nearest-neighbor examples from a training set as few-shot demos.
- `class KNN(Module)`
  - `.forward(**kwargs)` — embeds input, finds top-k similar training examples

#### `parallel.py`
Runs multiple DSPy modules concurrently in a thread pool.
- `class Parallel(Module)`
  - `.forward(modules, **kwargs)` — executes a list of modules in parallel

#### `rlm.py`
Reinforcement-learning-style module for online reward-based training updates.
- `class RLM(Module)`
  - `.forward(**kwargs)` — generates, scores, and records for online RL

#### `parameter.py`
- `class Parameter` — empty marker class; modules inheriting this are recognized by optimizers as tunable

---

### `primitives/`

#### `module.py`
Base class for all DSPy programs; the entry point for any user-defined AI system.
- `class Module(BaseModule)`
  - `.__call__(**kwargs)` / `.acall(**kwargs)` — invoke `forward` with callback hooks
  - `.named_predictors()` — yields `(name, Predict)` for every `Predict` sub-module
  - `.predictors()` — list of all `Predict` instances
  - `.set_lm(lm)` / `.get_lm()` — set/get the LM across all sub-predictors
  - `.map_named_predictors(func)` — apply a transformation to all predictors
  - `.batch(examples, num_threads)` — parallel batch processing via `Parallel`
  - `.inspect_history(n)` — print recent LM call history

#### `base_module.py`
Lower-level bookkeeping base; `Module` inherits from this.
- `class BaseModule`
  - `.named_parameters()` — recursive discovery of all `Parameter` sub-modules
  - `.dump_state() / load_state(state)` — delegate to each predictor's own state methods
  - `.save(path) / load(path)` — JSON serialization to disk
  - `.deepcopy() / reset_copy()` — copy with or without learned state

#### `example.py`
Lightweight dict-like container — the standard unit of training/eval data.
- `class Example`
  - `.with_inputs(*keys)` — mark which fields are inputs for optimization
  - `.inputs()` / `.labels()` — separate input and output fields
  - `.copy(**kwargs)` / `.without(*keys)` — immutable transformations
  - `.toDict()` — serialize to a plain dict

#### `prediction.py`
The output object returned by every `Predict`-derived module.
- `class Prediction(Example)`
  - `.from_completions(list_or_dict, signature)` — construct from raw LM outputs
  - `.completions` — property exposing the full `Completions` object
  - `.get_lm_usage() / set_lm_usage(value)` — token usage tracking
  - Arithmetic ops (`+`, `/`, comparisons) operate on a `.score` field
- `class Completions` — stores all N raw completions, indexable by position or field name

#### `python_interpreter.py`
Safe Python sandbox for executing LM-generated code.
- `class PythonInterpreter`
  - `.execute(code, context)` — runs code in a restricted namespace
  - `.reset()` — clear the execution state

#### `code_interpreter.py`
Higher-level wrapper orchestrating code execution for `ProgramOfThought`/`CodeAct`.
- `class CodeInterpreter`
  - `.execute(code)` — dispatch generated code to the right interpreter

#### `repl_types.py`
- `class CodePrompt` — typed wrapper for a block of code to be executed

---

### `adapters/`

#### `base.py`
Abstract base defining the prompt-formatting and output-parsing contract.
- `class Adapter`
  - `.__call__(lm, lm_kwargs, signature, demos, inputs)` — full pipeline: format → LM → parse
  - `.acall(...)` — async version
  - `.format(signature, demos, inputs)` — build the full message list
  - `.format_system_message(signature)` — instructions + field schema
  - `.format_field_description(signature)` — human-readable field descriptions
  - `.format_user_message_content(signature, inputs)` — format one user turn
  - `.format_assistant_message_content(signature, outputs)` — format one assistant turn
  - `.parse(signature, completion)` — extract typed field values from LM output

#### `chat_adapter.py`
- `class ChatAdapter(Adapter)` — formats as `[system, user, assistant, ...]` messages; default adapter

#### `json_adapter.py`
- `class JSONAdapter(Adapter)` — adds JSON schema to system prompt; parses JSON responses

#### `xml_adapter.py`
- `class XMLAdapter(Adapter)` — wraps fields in XML tags; parses `<field>value</field>` responses

#### `two_step_adapter.py`
- `class TwoStepAdapter(Adapter)` — step 1: free-form generation; step 2: structured extraction call

#### `baml_adapter.py`
- `class BAMLAdapter(Adapter)` — uses BAML DSL for structured extraction

#### `utils.py`
- `format_fields(fields, inputs)` — render field name/value pairs as prompt text
- `parse_value(value, type_)` — coerce a string to the declared Python type
- `get_annotation_name(annotation)` — pretty-print a type annotation

#### `types/`
Rich media input types used in multimodal signatures:
- `Image`, `Audio`, `File` — encode media as base64 or URL for LM input
- `History` — conversation history type for multi-turn modules
- `Tool`, `ToolCalls` — tool definitions and call records for agent modules
- `Code`, `Reasoning` — structured output types with native LM support

---

### `clients/`

#### `lm.py`
Primary LiteLLM-backed LM client — the main interface for making LM calls.
- `class LM(BaseLM)`
  - `.__init__(model, model_type, temperature, max_tokens, cache, ...)` — configure provider, model, and defaults
  - `.forward(prompt, messages, **kwargs)` — sync LM call with caching and retry
  - `.aforward(prompt, messages, **kwargs)` — async LM call
  - `.finetune(train_data, train_data_format, train_kwargs)` — trigger a fine-tuning job
  - `.reinforce(train_kwargs)` — RL-style weight update
  - `.dump_state() / load_state(state)` — serialize/deserialize LM config
  - `.launch(launch_kwargs)` / `.kill(launch_kwargs)` — start/stop local model servers

#### `base_lm.py`
- `class BaseLM` — abstract interface; all LM clients must implement `forward` and `aforward`

#### `lm_local.py`
- `class LocalLM(LM)` — runs models locally via vLLM or HuggingFace; extends `LM` with `launch`/`kill`

#### `openai.py`
- `class OpenAI(LM)` — OpenAI-specific client supporting structured outputs and the Responses API

#### `databricks.py`
- `class Databricks(LM)` — Databricks-specific client with Unity Catalog model routing

#### `embedding.py`
- `class Embedding` — client for embedding APIs (OpenAI, local, etc.)
  - `.__call__(inputs)` — returns embedding vectors for a list of strings

#### `cache.py`
Two-level (memory + disk) LRU cache that de-duplicates identical LM calls.
- `class Cache`
  - `.cache_key(request)` — deterministic SHA-256 hash of the request
  - `.get(request)` — check memory then disk
  - `.put(request, value)` — write to both levels
  - `.reset_memory_cache()` — clear the in-memory LRU
  - `.save_memory_cache(filepath)` / `.load_memory_cache(filepath)` — persist/restore memory cache
- `request_cache(...)` — decorator to apply `Cache` to any function

#### `provider.py`
- `infer_provider(model)` — detect LM provider (OpenAI, Anthropic, Cohere, …) from a model string

#### `utils_finetune.py`
- `format_finetune_data(traces, format)` — convert DSPy traces to JSON/JSONL for fine-tuning APIs
- `TrainDataFormat` — enum of supported fine-tuning data formats

---

### `teleprompt/` (Optimizers)

#### `teleprompt.py`
- `class Teleprompter` — abstract base; all optimizers implement `.compile(student, trainset, metric)`

#### `mipro_optimizer_v2.py`
State-of-the-art Bayesian optimizer for both instructions and few-shot demos.
- `class MIPROv2(Teleprompter)`
  - `.compile(student, trainset, metric, num_trials, ...)` — runs Bayesian optimization over proposed instructions

#### `bootstrap.py`
Collects passing traces from a teacher program and uses them as few-shot demos.
- `class BootstrapFewShot(Teleprompter)`
  - `.compile(student, trainset, teacher)` — run teacher, filter passing traces, attach as demos

#### `bootstrap_finetune.py`
Fine-tunes model weights using bootstrapped traces.
- `class BootstrapFinetune(Teleprompter)`
  - `.compile(student, trainset, metric)` — collect traces, format, call `LM.finetune`

#### `bettertogether.py`
Jointly alternates between prompt optimization and weight fine-tuning.
- `class BetterTogether(Teleprompter)`
  - `.compile(student, trainset, metric)` — alternating rounds of Bootstrap + Finetune

#### `grpo.py`
Group Relative Policy Optimization for RL-based LM fine-tuning.
- `class GRPO(Teleprompter)`
  - `.compile(student, trainset, metric)` — RL training loop using group reward normalization

#### `simba.py`
Stochastic mini-batch optimizer — fast hill-climbing over instruction/demo candidates.
- `class SIMBA(Teleprompter)`
  - `.compile(student, trainset, metric, num_steps, ...)` — iterative mini-batch swaps

#### `copro_optimizer.py`
Coordinate ascent — rewrites instructions for each module independently.
- `class COPRO(Teleprompter)`
  - `.compile(student, trainset, metric)` — LM-proposes new instructions module by module

#### `random_search.py`
- `class BootstrapFewShotWithRandomSearch(Teleprompter)`
  - `.compile(student, trainset, metric, num_candidates)` — random search over demo subsets

#### `knn_fewshot.py`
- `class KNNFewShot(Teleprompter)` — selects demos at runtime by embedding similarity to the current input

#### `ensemble.py`
- `class Ensemble(Teleprompter)` — combines multiple optimized programs via voting or stacking

#### `vanilla.py`
- `class LabeledFewShot(Teleprompter)` — simplest optimizer: directly attaches labeled examples as demos

#### `utils.py`
- `eval_candidate_program(program, devset, metric)` — parallelized scoring of a candidate
- `create_n_fewshot_demo_sets(trainset, num_sets)` — random sampling for demo generation

#### `gepa/`
GEPA (reflective prompt evolution), outperforming RL-based approaches per 2025 paper.
- `class GEPA(Teleprompter)` — candidate reflection + evolutionary selection loop

---

### `evaluate/`

#### `evaluate.py`
Parallelized evaluation runner over a dataset with a metric function.
- `class Evaluate`
  - `.__init__(devset, metric, num_threads, display_progress, ...)` — configure the evaluation
  - `.__call__(program, metric, display_table, ...)` — run and return `EvaluationResult`
  - `._construct_result_table(results, metric_name)` — build a pandas DataFrame of results
- `class EvaluationResult(Prediction)` — holds `.score` and `.results`

#### `metrics.py`
Ready-made string-matching metrics.
- `answer_exact_match(example, pred, frac)` — exact or F1-thresholded match
- `answer_passage_match(example, pred)` — checks if any passage contains any answer
- `EM(prediction, answers_list)` — multi-reference exact match
- `F1(prediction, answers_list)` — max token-level F1 over references
- `HotPotF1(prediction, answers_list)` — HotpotQA-style F1 with yes/no handling
- `normalize_text(s)` — lowercase, strip punctuation and articles
- `em_score / f1_score / hotpot_f1_score / precision_score` — low-level scoring functions

#### `auto_evaluation.py`
- `class SemanticF1` — LM-judged semantic similarity metric
- `LLMAsJudge` — wraps an LM call to return a 0/1 or float score for any prediction

---

### `retrievers/`

#### `retrieve.py`
- `class Retrieve(Module)` — base retriever module; queries the retrieval backend configured in `settings`
  - `.forward(query, k)` — return top-k passages

#### `embeddings.py`
- `class Embeddings(Module)` — in-memory cosine-similarity retriever
  - `.forward(query, k)` — embed query, retrieve top-k from embedded corpus

#### `databricks_rm.py`
- `class DatabricksRM(Module)` — retrieves from a Databricks Vector Search index or SQL table
  - `.forward(query, k)` — run vector search or SQL query, return passages

#### `weaviate_rm.py`
- `class WeaviateRM(Module)` — retrieves from a Weaviate vector database
  - `.forward(query, k)` — run Weaviate nearText query, return top-k passages

---

### `datasets/`

#### `dataset.py`
- `class Dataset` — base class for train/dev/test splits; provides `.train`, `.dev`, `.test`

#### `dataloader.py`
- `class DataLoader` — loads from HuggingFace Hub or local CSV/JSON files
  - `.from_huggingface(dataset_name, fields, ...)` — load as `Example` list
  - `.from_csv(file_path, fields)` — load from local CSV

#### `hotpotqa.py` / `gsm8k.py` / `math.py` / `colors.py`
- Each exposes a `HotPotQA()` / `GSM8K()` / `MATH()` / `Colors()` class with `.train` / `.dev` / `.test`

#### `alfworld/`
- `ALFWorld` — interactive household task environment with text-based observation/action spaces

---

### `streaming/`

#### `streamify.py`
- `streamify(program)` — wraps a DSPy program, returning an async generator of token chunks

#### `streaming_listener.py`
- `class StreamingListener` — subscribes to a specific `Predict` node's token stream
  - `.listen(predict_module)` — attach the listener
  - `.on_token(token)` — callback invoked per token

#### `messages.py`
- `class StatusMessage` — signals program lifecycle events (start, end, error)
- `class StreamResponse` — wraps a streamed token with metadata

---

### `propose/`

#### `grounded_proposer.py`
Generates new instruction candidates grounded in dataset summaries and examples.
- `class GroundedProposer`
  - `.propose_instructions_for_program(program, trainset, ...)` — returns a list of candidate instructions per module

#### `dataset_summary_generator.py`
- `create_dataset_summary(trainset, view_data_batch_size, program)` — calls an LM to summarize the dataset

#### `propose_base.py`
- `class Proposer` — abstract base; proposers implement `.propose_instructions_for_program`

#### `utils.py`
- `create_example_string(fields, example)` — format an `Example` as a string for inclusion in a proposal prompt
- `get_signature_name(signature)` — helper for labeling proposals by module

---

### `utils/`

#### `callback.py`
Hook system for instrumentation and observability.
- `class BaseCallback` — override any of: `on_lm_start`, `on_lm_end`, `on_module_start`, `on_module_end`
- `with_callbacks(func)` — decorator that fires registered callbacks around a function

#### `parallelizer.py`
- `class ParallelExecutor`
  - `.__init__(num_threads, max_errors, ...)` — configure thread pool
  - `.execute(fn, inputs)` — map `fn` over `inputs` in parallel, collect results

#### `asyncify.py`
- `asyncify(program)` — wraps a synchronous DSPy `Module` so it can be `await`-ed

#### `syncify.py`
- `syncify(program)` — runs an async DSPy `Module` from a synchronous context

#### `magicattr.py`
- `get(obj, path)` — deep attribute access via dotted path (e.g. `"predict.demos"`)
- `set(obj, path, value)` — deep attribute assignment via dotted path

#### `saving.py`
- `load(path)` — deserialize a saved DSPy program back to a live `Module` object

#### `usage_tracker.py`
- `track_usage()` — context manager; returns a dict of token counts after the block exits

#### `inspect_history.py`
- `inspect_history(n)` — pretty-print the last N LM prompt/response pairs

#### `hasher.py`
- `deterministic_hash(obj)` — SHA-256-based hash for cache-key generation

#### `callback.py`
- `configure_dspy_loggers(name)` — set up structured logging for the `dspy` logger
- `disable_logging()` / `enable_logging()` — toggle all DSPy log output

#### `exceptions.py`
- `class DSPyAssertionError` — raised by `dspy.Assert` when a constraint fails
- `class DSPyRetryError` — raised by `Retry` when all attempts are exhausted
- `class DSPySuggestionError` — soft constraint violation (non-fatal)

#### `dummies.py`
- `class DummyLM(BaseLM)` — configurable mock that returns preset answers without API calls
- `class DummyRM(Module)` — mock retriever returning fixed passages

#### `langchain_tool.py`
- `from_langchain_tool(tool)` — convert a LangChain `Tool` into a DSPy-compatible callable

#### `mcp.py`
- `from_mcp_tool(tool)` — convert an MCP (Model Context Protocol) tool into a DSPy-compatible callable

#### `unbatchify.py`
- `unbatchify(fn)` — wraps a batched LM call and expands results back to individual predictions

#### `caching.py`
- `enable_cache()` / `disable_cache()` — globally toggle DSPy's LM response cache

---

### `dsp/`

#### `colbertv2.py`
- `class ColBERTv2` — HTTP client for the ColBERT v2 dense retrieval server
  - `.__call__(query, k)` — return top-k passages from the ColBERT index

#### `utils/settings`
- `class Settings` — global singleton holding active LM, adapter, trace list, and context stack
  - `.configure(lm, adapter, ...)` — set global defaults
  - `.context(lm, adapter, ...)` — temporary override via a context manager
  - `settings` — the singleton instance imported everywhere as `dspy.settings`
