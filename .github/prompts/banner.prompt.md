---
description: This prompt is used to generate a wide Quarto blog-post banner illustration in a 19:4 aspect ratio, designed for HTML text overlay, with specific composition and style guidelines.
name: banner
model: GPT-4o
argument-hint: "<DESCRIPTION>, [TOPIC] , [HERO], [THEME], [PALETTE], [ARATIO], [FOLDER]"
---

If the `<DESCRIPTION>` argument is missing, default to `OSCD AI conference talk`.
If the `[TOPIC]` argument is missing, default to `ai`.
If the `[HERO] ` argument is missing, default to `a mascot like mythical creature`.
If the `[THEME]` argument is missing, default to `mosaic`.
If the `[PALETTE]` argument is missing, default to `deep navy, muted cyan, soft amber highlights, desaturated teal`.
If the `[ARATIO]` argument is missing, default to `19:4`.
If the `[FOLDER]` argument is missing, default to `images/banners`.

Create a wide Quarto blog-post banner illustration in a **[ARATIO] aspect ratio**.

The image must be designed for **HTML text overlay**, so it should contain **no written text, no labels, no captions, no numbers, no equations, no fake UI text, no logos, and no speaker headshots**.

Hero Element: [HERO]

Composition:

Left 20%: **[HERO]** a compact, visually memorable symbolic illustration representing: **[TOPIC]**. It should be attractive but not too bright or noisy.

Next 30%: a large muted title-safe area for overlaid Quarto title, subtitle, date, categories, and metadata. This area should be calm, dark, low-contrast, and mostly empty, with only very soft texture or atmospheric gradients. No arrows, sparks, dense lines, glowing objects, faces, or high-contrast details may cross this area.

Right 50%: the illustration may expand into a richer panoramic scene related to **[TOPIC]**, with more detail visible on large screens. Use visual metaphor rather than text: show concepts through objects, structure, motion, material, architecture, scientific instruments, abstract diagrams, or natural forms, but never through labels.

Theme: **[THEME]**

Style:

Elegant technical editorial illustration, cinematic but restrained, dark matte background, subtle depth, soft volumetric light, coherent color palette, high visual polish, no clutter. Use contrast mainly on the far left focal illustration and the far right extended scene; keep the title-safe region quiet and readable.

Avoid:

Fiery arrows, bright streaks through the center-left, dense circuitry under the title area, fake code windows with text, floating labels, speaker portraits, conference badges, logos, screenshots, diagrams with words, or any element that competes with overlaid typography.

Subject details for this banner:

**[DESCRIPTION]**

Preferred palette:

e.g. deep navy, muted cyan, soft amber highlights, desaturated teal**

**[palette]**

The final image should feel like a professional blog banner: readable under overlaid text, visually rich on wide screens, and meaningful without using any embedded words.

For your sparse-computing post, replace the subject line with something like:

**Sparse computing in the Python ecosystem: symbolic sparse tensors, compiler optimization, and hardware acceleration shown through abstract lattices, sparse matrix constellations, Python-like serpentine forms, and quiet computational geometry, without any text or labels.**

