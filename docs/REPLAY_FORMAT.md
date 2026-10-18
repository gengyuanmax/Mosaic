# Recorded trajectory collections

The demo loads `assets/replays/collections.json`, then the selected collection's
index, then individual episodes. A collection identifies one dataset, model and
evaluation setting. Adding another dataset or model does not require a new player.

## Register a prepared collection

Append one entry to `collections` with:

- `id`: unique public identifier; no internal experiment or checkpoint names.
- `dataset`, `model`, `setting`: each has a stable `id` and public `label`.
- `model.role_label`: optional shorter name for the conversation heading.
- `environment.label`: the environment actually used for the recorded run.
- `index`: path to that collection's index, relative to the site root.
- `attribution`, `license`: that collection's attribution and source terms.

Only register collections with prepared records and images. Do not expose an
unavailable dataset/model as a selectable placeholder. Distinct evaluation
settings must have separate collections. Repeated runs of the same configuration
can share one index, with distinct episode IDs. The Setting selector appears only
when the selected dataset/model has more than one setting.

## Collection index

Use `{ "default_id": "…", "episodes": […] }`. Each episode has `id`, `sample_id`,
`task`, `steps`, `tools`, and `file` (relative to the site root). Optional fields
include `thumbnail` and `transparent`. Keep `sample_id` stable within a dataset
across models so changing models can retain the same question when available.
Episode IDs identify recordings, not just questions. Image and JSON filenames
should be content-versioned when their contents change.

## Episode data

Follow the existing normalized episode format:

- `public_id`, `episode_id`, `task`, `question`, `ground_truth`.
- `final_answer`, `termination_reason`, `num_steps`.
- `rounds`: chronological recorded turns containing `model_raw_response`,
  `tool_calls`, `tool_results`, `final_answer`, `parse_error`, and `step`.
- Each tool call has `name`, `arguments`, `target_images`; pair it with exactly
  one result containing `new_image_ids`, feedback and any recorded error.
- `images`: stable workspace IDs, `source` parent IDs, `tool` (null for inputs),
  `file`, `thumbnail`, `full_file`, dimensions and `has_transparency`. Image paths
  are relative to `assets/`; all input images are shown beside the question.
- `credits`: source dataset, attribution, license/source links and modifications.

Keep the recorded prompt, image order, model text, answer and failures. Importers
for other record formats must normalize these fields without inventing reasoning,
tool calls or tool results. Answer-only recordings have no tool calls; the player
does not create an empty reasoning message for them. When reconstructing images,
use the corresponding recorded inputs and calls and retain their alpha channels.

## Loading and validation

The index cache holds four collections; the episode cache holds twelve records.
After selection, idle work fetches only adjacent episode JSON and small input
thumbnails within that collection. Full images load when needed. Slow/save-data
connections skip prefetching. New selections supersede older pending responses.

Before registering a collection, check IDs, question/answer mapping, paired tool
results, image references, original/reconstructed provenance, source terms and
absence of internal paths. Existing tests cover selector combinations, dataset
switch races, sample matching, recorded text and workspace timing.
