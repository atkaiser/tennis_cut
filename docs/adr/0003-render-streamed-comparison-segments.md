# Render streamed comparison segments

Status: superseded by ADR-0004

Each comparison clip is rendered directly through FFmpeg from its user source and a cached, lossless prepared pro panel, then the encoded segments are concatenated without re-encoding. This avoids full-resolution image intermediates while keeping progress, failures, optional clip publication, and atomic staging simpler than one filter graph spanning every swing.
