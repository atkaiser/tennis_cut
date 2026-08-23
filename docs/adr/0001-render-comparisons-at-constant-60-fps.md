# Render comparisons at constant 60 fps

Comparison artifacts use a constant 60 fps timeline rather than preserving the exact variable-frame event union of their sources. This permits direct FFmpeg stream composition and predictable playback while retaining exact alignment of the selected contact frames; preserving every source presentation timestamp is not a product requirement.
