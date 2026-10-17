"""Validate a sample matches the canonical schema (PREPROCESSING, not runtime).

Run as part of converting/QA-ing a raw dataset into the standardized
``data/mmiu/datasets/<task>.json``. Not imported by the runtime package.
"""

from canonical import REQUIRED_FIELDS  # sibling module (run from scripts/data_prep/)


def validate_canonical_sample(sample):
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in sample:
            errors.append("missing required field: %s" % field)

    if "prompt" in sample and not sample["prompt"]:
        errors.append("prompt must be non-empty")

    if "images" in sample:
        if not isinstance(sample["images"], list) or len(sample["images"]) == 0:
            errors.append("images must be a non-empty list")
        else:
            seen = set()
            for idx, image in enumerate(sample["images"]):
                image_id = image.get("image_id") if isinstance(image, dict) else None
                path = image.get("uri_or_path") if isinstance(image, dict) else None
                if not image_id:
                    errors.append("images[%d].image_id is required" % idx)
                if image_id in seen:
                    errors.append("images[%d].image_id is duplicated" % idx)
                seen.add(image_id)
                if not path:
                    errors.append("images[%d].uri_or_path is required" % idx)

    return len(errors) == 0, errors
