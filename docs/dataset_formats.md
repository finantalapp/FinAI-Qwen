# Dataset Formats

The project accepts data in several common formats and **auto-detects** each
record, normalising everything to one unified shape before training. Working
examples for each format live in [`data/samples/`](../data/samples).

## Unified shape

Every record becomes either:

- a **chat** record — `{"messages": [{"role", "content"}, ...]}`, or
- a **plain-text** record — `{"text": "..."}` (for continued-pretraining data).

The chat shape is exactly what `tokenizer.apply_chat_template` consumes, so the
trainer never branches on the original format.

Valid roles: `system`, `user`, `assistant` (and `tool`).

## Supported input formats

### OpenAI messages (`openai`)
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```
Already unified. `content` may also be a list of `{"type": "text", "text": ...}`
parts, which are concatenated.

### ShareGPT (`sharegpt`)
```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```
`from` is mapped to a role: `human/user → user`, `gpt/assistant/bot/... →
assistant`, `system → system`. A top-level `system` field is honoured.

### Alpaca (`alpaca`)
```json
{"instruction": "...", "input": "(optional)", "output": "...", "system": "(optional)"}
```
`instruction` (+ `input`, joined by a blank line) becomes the user turn; `output`
becomes the assistant turn.

### ChatML (`chatml`)
```json
{"text": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"}
```
The `<|im_start|>role ... <|im_end|>` turns are parsed into messages.

### Prompt / completion (`prompt`)
```json
{"prompt": "...", "completion": "..."}      // or "response" instead of "completion"
```

### Plain text (`text`)
```json
{"text": "raw text for continued pretraining"}
```

## File types

- `.jsonl` — one JSON record per line (recommended).
- `.json` — either a JSON array of records or a single record object.
- A **directory** — all `.jsonl`/`.json` files under it are read recursively and
  concatenated.

## How loading works

`src/data/loader.py::prepare_dataset`:

1. resolves the path (absolute → repo-relative → `datasets/`-relative),
2. reads raw records,
3. normalises each via `src/data/formats.py::normalize_record`,
4. renders each through the chat template into a `text` column (plain-text
   records pass through verbatim),
5. drops empty/invalid examples (a usable chat example needs a user turn and a
   non-empty assistant turn),
6. splits into train/eval by `eval_split_ratio`.

Set the format explicitly with `--format` (or `dataset_format`) to skip
auto-detection; `auto` (default) detects per record, so **mixed-format files are
supported**.

## Instruct vs base models

Chat-format data is rendered with the tokenizer's chat template. **Base** models
(e.g. `Qwen/Qwen2.5-7B` without `-Instruct`) may not ship a template; in that
case either:

- use an `-Instruct` model (recommended for chat/SFT), or
- provide **plain-text** data (`{"text": ...}`), or
- set a `chat_template` on the tokenizer before training.

## Adding a new format

1. Add a converter and a detection branch in `src/data/formats.py`.
2. Add it to the `FormatName` literal and the `normalize_record` dispatch.
3. Add a unit test in `tests/test_formats.py` and a sample in `data/samples/`.
4. Document it here and in `docs/changelog.md`.
