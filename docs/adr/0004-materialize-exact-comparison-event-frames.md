# Materialize exact comparison event frames

Comparison rendering decodes every planned user and professional source-frame ordinal to an image, composes each event frame with Pillow, and performs one variable-frame-rate encode over that materialized sequence. Each source is decoded once for the complete requested artifact set. Each distinct source-frame-and-crop combination is resized into a panel once; the reusable professional panels and user panels then feed the compilation and any requested individual clips.

This remains slower than streaming segments directly through FFmpeg, but it preserves an explicit one-to-one mapping between the planner's selected contact frame and the image used at that event. The streamed approach introduced a real-video contact-alignment regression and is superseded. Sharing decoded frames and prepared panels changes their lifetime, not the exact-frame selection or materialized-event model.
