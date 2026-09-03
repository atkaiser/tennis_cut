# Materialize exact comparison event frames

Comparison rendering decodes every planned user and professional source-frame ordinal to an image, composes each event frame with Pillow, and performs one variable-frame-rate encode over that materialized sequence. This is slower than streaming segments directly through FFmpeg, but it preserves an explicit one-to-one mapping between the planner's selected contact frame and the image used at that event; the streamed approach introduced a real-video contact-alignment regression and is superseded.
