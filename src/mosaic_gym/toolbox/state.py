"""Image state — the per-trajectory image store the tools operate on.

Held by one ``Toolbox`` instance (= one rollout). Tools read/write images by id;
``crop`` produces ``img_0005`` and a later ``diff`` consumes it from the same
store. The prompt is kept here too (no current tool reads it — the labelled
matrix_ref/transform_ref resolvers were removed)."""

from dataclasses import dataclass, field
from typing import Dict, List

from PIL import Image


@dataclass
class ImageNode:
    image_id: str
    image: Image.Image


@dataclass
class ImageState:
    images: Dict[str, ImageNode] = field(default_factory=dict)
    prompt: str = ""  # kept for context; no current tool reads it

    def add_image(self, image: Image.Image) -> str:
        new_id = "img_%04d" % (len(self.images) + 1)
        # Store-wide invariant: everything in the store is RGBA, so alpha
        # (transparent = no content) survives between tools. ``convert``
        # also returns a new image, preserving the old defensive copy.
        self.images[new_id] = ImageNode(image_id=new_id, image=image.convert("RGBA"))
        return new_id

    def get_image(self, image_id: str) -> Image.Image:
        """Return the PIL image for ``image_id`` — a SHARED reference, not a copy.
        Callers (tools) must not mutate it in place; ``.copy()`` first, or call
        ``add_image`` to register a derived image and return its new id."""
        if image_id not in self.images:
            raise KeyError(image_id)
        return self.images[image_id].image

    def image_ids(self) -> List[str]:
        """Every image id now in the store, in creation order (img_0001, img_0002,
        …). The agent maps these in order to the model-facing [imgN] tokens."""
        return sorted(self.images.keys())

    def images_by_id(self) -> Dict[str, Image.Image]:
        """``{image_id: PIL}`` for every image now in the store — the runner reads
        this to hand the agent the actual pixels to render (base64)."""
        return {iid: node.image for iid, node in self.images.items()}


def coerce_image(value):
    # RGBA so alpha survives intake: transparent pixels (rotate corners,
    # collage gaps) must reach the tools as alpha=0, not baked-in black.
    if isinstance(value, Image.Image):
        return value.convert("RGBA")
    if isinstance(value, str):
        return Image.open(value).convert("RGBA")
    raise TypeError("unsupported image input type: %s" % type(value))
