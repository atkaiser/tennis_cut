"""Rebuild prototype training records from the retained detection pilot."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tennis_cut.visual_contact import Detection, FrameEvidence, VisualFrame, extract_temporal_features


def family(source: str) -> str:
    match = re.search(r"IMG_(\d+)", Path(source).stem)
    if match is None:
        return Path(source).stem
    ordinal = int(match.group(1))
    for name, lower, upper in (("857", 8570, 8579), ("861", 8610, 8619), ("863", 8630, 8649), ("867", 8670, 8679), ("911", 9110, 9119), ("912", 9120, 9139)):
        if lower <= ordinal <= upper:
            return name
    return Path(source).stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, default=Path("out/contact-frame-prototype"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads((args.pilot / "results.json").read_text())
    records = []
    for result in results:
        source = Path(result["source"])
        candidates = list((args.pilot / "detections").glob(f"{source.stem}-{result['manual_frame']}-*.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected one cache for {source} frame {result['manual_frame']}, found {len(candidates)}")
        payload = json.loads(candidates[0].read_text())
        frames = tuple(
            VisualFrame(FrameEvidence(
                int(frame["ordinal"]),
                0,
                tuple(Detection(
                    {0: "person", 32: "ball", 38: "racket"}[int(detection["class_id"])],
                    tuple(float(value) for value in detection["box"]),
                    float(detection["confidence"]),
                ) for detection in frame["detections"] if int(detection["class_id"]) in {0, 32, 38}),
            ))
            for frame in payload["frames"]
        )
        features = extract_temporal_features(frames)
        records.append({
            "group": family(result["source"]),
            "label_frame": int(result["manual_frame"]),
            # results.json stores the post-temporal selection. The first
            # deterministic ranking entry and only the pre-temporal omission
            # reasons are needed to reproduce prototype calibration.
            "deterministic_frame": (int(result["top_ranking"][0]["frame"]) if result["top_ranking"] else None),
            "omission_reason": result["omission_reason"] if result["omission_reason"] in {"weak visual evidence", "broad or separated ambiguity"} else None,
            "features": [{"frame_ordinal": feature.frame_ordinal, "values": list(feature.values)} for feature in features],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
