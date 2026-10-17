"""Tool framework: the tool interface (ToolBase), the registry, and argument validation."""

import math
from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Dict, Iterable, List, Tuple

from mosaic_gym.contracts import ToolOutput


# Allowed values for ToolBase.feedback_form. Per-tool declaration of what the
# model sees: just the produced image (default), image + text, or text only
# (a judgment tool that returns no image). There is no global override — a
# text-bearing tool just sets its own feedback_form. (No tool currently uses
# text_only; the form is kept for future judgment tools.)
FEEDBACK_FORM_IMAGE_ONLY = "image_only"
FEEDBACK_FORM_IMAGE_AND_TEXT = "image_and_text"
FEEDBACK_FORM_TEXT_ONLY = "text_only"
SUPPORTED_FEEDBACK_FORMS = frozenset({
    FEEDBACK_FORM_IMAGE_ONLY,
    FEEDBACK_FORM_IMAGE_AND_TEXT,
    FEEDBACK_FORM_TEXT_ONLY,
})


def _validator_schema_from(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Derive the validator-facing schema from the canonical :meth:`ToolBase.schema`.

    The schema separates two interfaces: the canonical ``schema()`` includes the
    tool's image arguments in properties + required (so the prompt renderer
    mentions them and the structured-output JSON Schema forces the model to
    emit them inside ``tool.arguments``), while the runtime extracts them out
    of ``tool.arguments`` into ``ToolCall.target_images`` before invoking the
    registry validator — so by the time ``validate_tool_arguments`` runs, the
    arguments dict no longer contains them. which properties are
    image arguments is declared per property via the internal
    ``"image_ref": True`` marker (previously the hardcoded ``target_images``
    name), so strip every marked property and its required entry here to
    avoid a spurious missing-argument error.
    """
    props = {
        k: v
        for k, v in (schema.get("properties") or {}).items()
        if not (isinstance(v, dict) and v.get("image_ref"))
    }
    required = [k for k in (schema.get("required") or []) if k in props]
    return {"required": required, "properties": props}


class ToolBase(metaclass=ABCMeta):
    name = "base"
    # Two description tiers, selected by the toolbox config's ``description_mode``:
    #   baseline   — 1 sentence: WHAT the tool does + key arg format. No policy.
    #   prompting  — baseline + a short "Use when … ; avoid when …" steer.
    # The mode lets us A/B "let the model figure out usage" (baseline) vs
    # "guide it" (prompting). ``description`` is the legacy single blob, kept as a
    # fallback for any tool that hasn't split yet.
    description_baseline = ""
    description_prompting = ""
    description = ""
    min_target_images = 1
    max_target_images = 1

    # Per-tool feedback contract. Most tools' output IS the produced image,
    # so the default is ``image_only``: the model sees only the image next turn
    # (the tool may still set tool_feedback_text for the record, but it is not
    # surfaced to the model). A judgment tool with no image would override this
    # to ``text_only``; a tool whose numbers can't be read off the image can
    # use ``image_and_text``.
    feedback_form: ClassVar[str] = FEEDBACK_FORM_IMAGE_ONLY

    # ------------------------------------------------------------------
    # single source of truth for tool argument shape.
    #
    # Pre-#27 every tool maintained two near-identical JSON-Schema
    # fragments: ``arg_schema()`` (validator-facing, no descriptions, no
    # target_images entry) and ``model_arg_schema()`` (model-facing, full
    # descriptions + target_images entry). They had to be kept in sync by
    # hand; PR-A1 #23 added a bidirectional test that caught drift in
    # *names* but not in ``required`` lists (CollageTool drift slipped
    # through). #27 collapses both into ``schema()``: tools declare the
    # rich, model-facing shape once; the validator-facing version is
    # derived automatically by :func:`_validator_schema_from`.
    #
    # ``arg_schema`` and ``model_arg_schema`` are kept for one release as
    # deprecated aliases that delegate to ``schema()`` so existing
    # external callers (and any not-yet-migrated tool) keep working.
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def schema(cls) -> Dict[str, Any]:
        """Single source of truth for this tool's argument shape.

        Returns a dict shaped like::

            {
                "required": [<arg_name>, ...],
                "properties": {
                    "<arg_name>": {
                        "type": ...,
                        "description": "...",
                        ...constraints (minimum / maximum / enum / ...),
                    },
                    "target_images": {
                        "type": "array",
                        "image_ref": True,  # internal marker: image argument
                        "items": {"type": "string"},
                        "description": "...",
                    },
                },
            }

        The agent's tool-schema builder (``agent.build_tool_schemas``)
        and the argument validator (``validate_tool_arguments`` below)
        both derive from this one method. Descriptions are required on
        every property except where omitted by intent.
        """
        raise NotImplementedError

    @classmethod
    def arg_schema(cls) -> Dict[str, Any]:
        """Deprecated — use :meth:`schema` instead.

        Returns the validator-facing view of :meth:`schema` (with the
        ``image_ref``-marked entries stripped, since the runtime extracts
        the image arguments before invoking the validator). Kept for one
        release so external callers and any pre-#27 tool subclass that
        only overrides ``arg_schema`` keep working transparently.
        """
        return _validator_schema_from(cls.schema())

    @classmethod
    def model_arg_schema(cls) -> Dict[str, Any]:
        """Deprecated — use :meth:`schema` instead.

        Returns :meth:`schema` directly (the model-facing view IS the
        canonical schema). Kept for one release for external callers.
        """
        return cls.schema()

    def description_for(self, description_mode: str = "prompting") -> str:
        """Pick the description for the active mode, falling back across tiers."""
        if description_mode == "baseline":
            return self.description_baseline or self.description_prompting or self.description
        return self.description_prompting or self.description_baseline or self.description

    def spec(self, description_mode: str = "prompting") -> Dict[str, Any]:
        # ``arg_schema`` / ``model_arg_schema`` kept here for back-compat
        # with downstream consumers that index into the spec dict
        # directly. New code should read ``schema`` instead.
        canonical = type(self).schema()
        return {
            "name": str(self.name),
            "description": str(self.description_for(description_mode)),
            "min_target_images": int(self.min_target_images),
            "max_target_images": (
                None if self.max_target_images is None else int(self.max_target_images)
            ),
            "schema": canonical,
            "arg_schema": _validator_schema_from(canonical),
            "model_arg_schema": canonical,
        }

    def validate_arguments(self, arguments: Dict[str, Any]) -> List[str]:
        _ = arguments
        return []

    @abstractmethod
    def execute(self, state, arguments: Dict[str, Any], target_images: List[str]) -> ToolOutput:
        raise NotImplementedError


# ======================================================================
# Registry — the collection of tools, keyed by name.
# ======================================================================


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, tool):
        if not isinstance(tool, ToolBase):
            raise TypeError("tool must implement ToolBase")
        self._tools[tool.name] = tool

    def register_many(self, tools: Iterable[ToolBase]):
        for tool in tools:
            self.register(tool)

    def get(self, name: str):
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self, description_mode: str = "prompting") -> List[Dict[str, Any]]:
        result = []
        for name in sorted(self._tools.keys()):
            tool = self._tools[name]
            result.append(tool.spec(description_mode))
        return result


# ======================================================================
# Argument validation — check a model-supplied arguments dict against a
# tool's JSON-Schema-ish ``schema()`` and return clear, model-facing errors.
# ======================================================================


def validate_tool_arguments(schema: Dict[str, Any], arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for key in required:
        if key not in arguments:
            errors.append("missing required argument: %s" % key)

    for key, value in arguments.items():
        if key not in properties:
            errors.append("unknown argument: %s" % key)
            continue
        spec = properties[key]
        spec_type = spec.get("type")
        if not _is_valid_type(spec_type, value):
            errors.append("argument %s must be %s" % (key, spec_type))
            continue

        if "enum" in spec and value not in spec["enum"]:
            errors.append("argument %s must be one of %s" % (key, spec["enum"]))

        if spec_type in ("integer", "number"):
            # NaN / +inf / -inf bypass all bound checks (Python's < / > are
            # always False for NaN, and inf slips past any unbounded side).
            # Reject non-finite values up front with a clean error.
            if isinstance(value, float) and not math.isfinite(value):
                errors.append("argument %s must be a finite number, got %r" % (key, value))
                continue
            if "minimum" in spec and value < spec["minimum"]:
                errors.append("argument %s must be >= %s" % (key, spec["minimum"]))
            if "exclusiveMinimum" in spec and value <= spec["exclusiveMinimum"]:
                errors.append("argument %s must be > %s" % (key, spec["exclusiveMinimum"]))
            if "maximum" in spec and value > spec["maximum"]:
                errors.append("argument %s must be <= %s" % (key, spec["maximum"]))
            if "exclusiveMaximum" in spec and value >= spec["exclusiveMaximum"]:
                errors.append("argument %s must be < %s" % (key, spec["exclusiveMaximum"]))

        if spec_type == "array":
            if "minItems" in spec and len(value) < spec["minItems"]:
                errors.append("argument %s must contain at least %s items" % (key, spec["minItems"]))
            if "maxItems" in spec and len(value) > spec["maxItems"]:
                errors.append("argument %s must contain at most %s items" % (key, spec["maxItems"]))
            items_spec = spec.get("items") or {}
            item_type = items_spec.get("type")
            for idx, item in enumerate(value):
                if item_type and not _is_valid_type(item_type, item):
                    errors.append("argument %s[%d] must be %s" % (key, idx, item_type))
                    continue
                if item_type in ("integer", "number"):
                    if isinstance(item, float) and not math.isfinite(item):
                        errors.append(
                            "argument %s[%d] must be a finite number, got %r" % (key, idx, item)
                        )
                        continue
                    if "minimum" in items_spec and item < items_spec["minimum"]:
                        errors.append("argument %s[%d] must be >= %s" % (key, idx, items_spec["minimum"]))
                    if "exclusiveMinimum" in items_spec and item <= items_spec["exclusiveMinimum"]:
                        errors.append("argument %s[%d] must be > %s" % (key, idx, items_spec["exclusiveMinimum"]))
                    if "maximum" in items_spec and item > items_spec["maximum"]:
                        errors.append("argument %s[%d] must be <= %s" % (key, idx, items_spec["maximum"]))
                    if "exclusiveMaximum" in items_spec and item >= items_spec["exclusiveMaximum"]:
                        errors.append("argument %s[%d] must be < %s" % (key, idx, items_spec["exclusiveMaximum"]))

    return len(errors) == 0, errors


_SUPPORTED_JSON_TYPES = {"integer", "number", "string", "array", "object", "boolean", "null"}


def _is_valid_type(expected: str, value: Any) -> bool:
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ValueError(
        "tool schema declared unknown type %r; supported: %s"
        % (expected, sorted(_SUPPORTED_JSON_TYPES))
    )
