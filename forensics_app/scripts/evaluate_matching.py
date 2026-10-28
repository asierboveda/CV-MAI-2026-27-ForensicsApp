"""Evaluate folder matching.

Run from the repository:
    uv run forensics_app/scripts/evaluate_matching.py
"""

import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import ORB, SIFT, match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform


IMAGE_FOLDER = (
    Path(__file__).resolve().parents[1]
    / "images/Set5_images/buildings"
)

# Use validation cases to choose the configuration.
# Keep both directions of each image pair in the same split.
VALIDATION_CASES = {
    "ciudad_ciencias_1.jpg": {"ciudad_ciencias_2.jpg"},
    "ciudad_ciencias_2.jpg": {"ciudad_ciencias_1.jpg"},
    "ontario1.jpg": {"ontario2.jpeg"},
    "ontario2.jpeg": {"ontario1.jpg"},
}

# Evaluate the selected configuration on a different landmark.
TEST_CASES = {
    "eiffel_tower1.jpg": {"eiffel_tower2.jpg"},
    "eiffel_tower2.jpg": {"eiffel_tower1.jpg"},
}

# name: (descriptor, maximum image dimension, RANSAC threshold)
CONFIGURATIONS = {
    "ORB original": ("ORB", None, None),
    "ORB resized": ("ORB", 1000, None),
    "SIFT resized": ("SIFT", 1000, None),
    "ORB + RANSAC": ("ORB", 1000, 3.0),
    "SIFT + RANSAC": ("SIFT", 1000, 3.0),
}

MAX_RATIO = 0.8

# A provisional rejection threshold, not a guarantee of a correct match.
MIN_INLIERS = 8


def extract_features(path, method, max_size):
    with Image.open(path) as image:
        image = image.convert("RGB")

    if max_size is not None:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    gray = rgb2gray(np.asarray(image))
    if method == "SIFT" and min(gray.shape) < 6:
        return np.empty((0, 2)), None
    extractor = ORB(n_keypoints=500) if method == "ORB" else SIFT()

    try:
        extractor.detect_and_extract(gray)
    except RuntimeError:
        return np.empty((0, 2)), None

    # Convert (row, column) to (x, y) for geometric transforms.
    return extractor.keypoints[:, ::-1].copy(), extractor.descriptors.copy()


def compare_features(reference, candidate, ransac_threshold):
    points1, descriptors1 = reference
    points2, descriptors2 = candidate

    if descriptors1 is None or descriptors2 is None:
        return 0, 0

    # The ratio test needs a second candidate descriptor.
    if len(descriptors2) < 2:
        return 0, 0

    matches = match_descriptors(
        descriptors1,
        descriptors2,
        cross_check=True,
        max_ratio=MAX_RATIO,
    )

    match_count = len(matches)

    if ransac_threshold is None or match_count < 4:
        return match_count, 0

    source = points1[matches[:, 0]]
    destination = points2[matches[:, 1]]

    try:
        model, inliers = ransac(
            (source, destination),
            ProjectiveTransform,
            min_samples=4,
            residual_threshold=ransac_threshold,
            max_trials=1000,
            rng=0,
        )
    except (ValueError, np.linalg.LinAlgError):
        return match_count, 0

    if model is None or inliers is None:
        return match_count, 0

    return match_count, int(inliers.sum())


def evaluate_configuration(name, configuration, cases, paths, split):
    method, max_size, ransac_threshold = configuration
    start = perf_counter()

    # Extract each image once. Searching does not generate plots.
    features = {
        path.name: extract_features(path, method, max_size)
        for path in paths
    }
    extraction_seconds = perf_counter() - start

    rows = []

    for reference_name, expected in cases.items():
        search_start = perf_counter()
        candidates = []

        for path in paths:
            if path.name == reference_name:
                continue

            matches, inliers = compare_features(
                features[reference_name],
                features[path.name],
                ransac_threshold,
            )

            score = matches if ransac_threshold is None else inliers
            candidates.append((path.name, score, matches, inliers))

        # Paths are sorted, so ties keep the first filename.
        best = max(candidates, key=lambda candidate: candidate[1])
        winner, score, matches, inliers = best

        minimum_score = 1 if ransac_threshold is None else MIN_INLIERS
        if score < minimum_score:
            winner = ""

        rows.append({
            "split": split,
            "configuration": name,
            "reference": reference_name,
            "expected": ";".join(sorted(expected)),
            "winner": winner,
            "correct": winner in expected,
            "matches": matches,
            "inliers": inliers,
            "score": score,
            "search_seconds": round(perf_counter() - search_start, 4),
        })

    accuracy = sum(row["correct"] for row in rows) / len(rows)
    total_seconds = perf_counter() - start

    summary = {
        "configuration": name,
        "accuracy": accuracy,
        "extraction_seconds": round(extraction_seconds, 4),
        "total_seconds": round(total_seconds, 4),
    }

    return rows, summary


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(
        path
        for path in IMAGE_FOLDER.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )

    results = []
    summaries = []

    print("VALIDATION")
    print(f"{'Configuration':<20} {'Accuracy':>10} {'Total seconds':>15}")

    for name, configuration in CONFIGURATIONS.items():
        rows, summary = evaluate_configuration(
            name, configuration, VALIDATION_CASES, paths, "validation"
        )
        results.extend(rows)
        summaries.append(summary)

        print(
            f"{name:<20} "
            f"{summary['accuracy']:>9.0%} "
            f"{summary['total_seconds']:>15.2f}"
        )

    # Prefer accuracy; use runtime to break ties.
    best = max(
        summaries,
        key=lambda summary: (
            summary["accuracy"],
            -summary["total_seconds"],
        ),
    )
    best_name = best["configuration"]

    print(f"\nSelected configuration: {best_name}")
    print("\nTEST")

    test_rows, test_summary = evaluate_configuration(
        best_name,
        CONFIGURATIONS[best_name],
        TEST_CASES,
        paths,
        "test",
    )
    results.extend(test_rows)

    for row in test_rows:
        winner = row["winner"] or "No reliable match"
        status = "OK" if row["correct"] else "FAIL"
        print(f"{row['reference']} -> {winner} [{status}]")

    print(f"\nTest accuracy: {test_summary['accuracy']:.0%}")

    save_csv(Path("matching_results.csv"), results)
    save_csv(Path("matching_summary.csv"), summaries)

    print("Saved matching_results.csv and matching_summary.csv")


if __name__ == "__main__":
    main()
