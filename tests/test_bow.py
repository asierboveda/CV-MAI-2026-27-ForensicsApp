import unittest
from pathlib import Path
import tkinter as tk
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.bow_localize import VisualWordLocalizationWindow
from forensics_app.tools.bow_utils import (
    MODELS_DIR,
    build_bow_histogram,
    compute_histogram_similarities,
    extract_keypoints_and_assignments,
    get_active_model_dir,
    load_classes,
    load_classifier,
    load_kmeans,
    load_train_histograms,
    load_train_image_records,
    load_visual_words,
    load_visual_words_grid,
    render_histogram,
    render_localization_overlay,
    render_localization_view,
    render_retrieval_composite,
    render_single_image_histogram_with_top_words,
    set_active_model_dir,
    validate_model_dir,
)


class BoWTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.image = Image.fromarray(arr)
        self.document = ImageDocument()
        self.document.current = self.image

    def test_bow_tools_registered(self):
        tools = {t.tool_id: t for t in build_tool_registry().all()}
        self.assertIn("bow_build", tools)
        self.assertIn("bow_histogram", tools)
        self.assertIn("bow_classify", tools)
        self.assertIn("bow_retrieval", tools)
        self.assertNotIn("bow_visual_words", tools, "Visualise Codebook is integrated inside Construct Codebook")
        self.assertNotIn("bow_localize", tools, "Spatial localization is accessible via BoW Histogram")
        self.assertEqual(tools["bow_build"].category, "Set 7")
        self.assertEqual(tools["bow_histogram"].category, "Set 7")
        self.assertEqual(tools["bow_classify"].category, "Set 7")
        self.assertEqual(tools["bow_retrieval"].category, "Set 7")

    def test_codebook_directory_validation_and_switching(self):
        valid, msg = validate_model_dir(MODELS_DIR)
        self.assertTrue(valid)

        # Non-existent directory
        fake_dir = Path("non_existent_folder_xyz")
        valid_fake, _ = validate_model_dir(fake_dir)
        self.assertFalse(valid_fake)

        # Set and get active model dir
        orig = get_active_model_dir()
        set_active_model_dir(MODELS_DIR)
        self.assertEqual(get_active_model_dir(), MODELS_DIR.resolve())

    def test_codebook_k_range_validation(self):
        from forensics_app.tools.bow_utils import DATASET_DIR, build_codebook_bundle
        if DATASET_DIR.is_dir():
            # K must be > 1
            with self.assertRaises(ValueError):
                build_codebook_bundle(DATASET_DIR, Path("dummy"), k=1)
            # K must be < total images
            with self.assertRaises(ValueError):
                build_codebook_bundle(DATASET_DIR, Path("dummy"), k=9999)

    def test_minibatch_kmeans_compatibility(self):
        from sklearn.cluster import MiniBatchKMeans
        # Test that MiniBatchKMeans works seamlessly with build_bow_histogram
        np.random.seed(42)
        dummy_descs = np.random.randn(100, 128).astype(np.float64)
        mb_kmeans = MiniBatchKMeans(n_clusters=5, batch_size=32, random_state=42).fit(dummy_descs)
        hist = build_bow_histogram(self.image, mb_kmeans)
        self.assertEqual(len(hist), 5)
        self.assertAlmostEqual(hist.sum(), 1.0, places=4)

    def test_render_histogram(self):
        hist = np.ones(30, dtype=np.float64) / 30.0
        chart = render_histogram(hist, title="Test Chart", annotation="p(test) = 0.99")
        self.assertIsInstance(chart, Image.Image)
        self.assertEqual(chart.mode, "RGB")
        self.assertGreater(chart.width, 0)
        self.assertGreater(chart.height, 0)

    def test_render_single_image_histogram_with_top_words(self):
        hist = np.zeros(30, dtype=np.float64)
        hist[4] = 0.4
        hist[2] = 0.3
        hist[8] = 0.15
        hist[1] = 0.1
        hist[10] = 0.05
        records = load_visual_words()
        composite, top5_info = render_single_image_histogram_with_top_words(
            self.image, hist, records, image_name="Test Image"
        )
        self.assertIsInstance(composite, Image.Image)
        self.assertEqual(len(top5_info), 5)
        self.assertEqual(top5_info[0][1], 4)  # Rank 1 is vw 4
        self.assertEqual(top5_info[1][1], 2)  # Rank 2 is vw 2

    def test_models_exist_and_load(self):
        kmeans = load_kmeans()
        self.assertEqual(kmeans.n_clusters, 30)

        clf = load_classifier()
        self.assertIsNotNone(clf)

        X_train = load_train_histograms()
        self.assertEqual(X_train.shape[1], 30)
        self.assertEqual(X_train.shape[0], 193)

        classes = load_classes()
        self.assertEqual(classes, ["butterfly", "dalmatian", "kangaroo"])

        grid = load_visual_words_grid()
        self.assertIsInstance(grid, Image.Image)

        records = load_train_image_records()
        self.assertEqual(len(records), 193)
        self.assertTrue(all(p.exists() for p, _ in records[:5]))

    def test_localization_helpers(self):
        kps = np.array([[20.0, 30.0], [45.0, 60.0], [80.0, 10.0]])
        assignments = np.array([4, 2, 4])
        overlay, stats = render_localization_overlay(
            self.image, kps, assignments, selected_words=[4, 2]
        )
        self.assertIsInstance(overlay, Image.Image)
        self.assertEqual(overlay.size, self.image.size)
        self.assertEqual(stats[4]["count"], 2)
        self.assertEqual(stats[2]["count"], 1)

        records = load_visual_words()
        view = render_localization_view(
            self.image, kps, assignments, top_words=[4, 2], visual_words_records=records
        )
        self.assertIsInstance(view, Image.Image)

    def test_cbir_helpers(self):
        X_train = load_train_histograms()
        q_hist = np.ones(30, dtype=np.float64) / 30.0
        scores_chi2 = compute_histogram_similarities(q_hist, X_train, metric="chi2")
        self.assertEqual(scores_chi2.shape, (193,))
        self.assertTrue(np.all((scores_chi2 >= 0.0) & (scores_chi2 <= 1.0)))

        scores_inter = compute_histogram_similarities(q_hist, X_train, metric="intersection")
        self.assertEqual(scores_inter.shape, (193,))
        self.assertTrue(np.all((scores_inter >= 0.0) & (scores_inter <= 1.0)))

        records = load_train_image_records()
        top_matches = [
            {"rank": 1, "path": records[0][0], "class": records[0][1], "similarity": 0.95},
            {"rank": 2, "path": records[1][0], "class": records[1][1], "similarity": 0.88},
        ]
        composite = render_retrieval_composite(
            self.image, "test.png", q_hist, top_matches, metric_name="Chi-squared"
        )
        self.assertIsInstance(composite, Image.Image)

    def test_tools_run_headless(self):
        root = tk.Tk()
        root.withdraw()
        try:
            tools = {t.tool_id: t for t in build_tool_registry().all()}

            # Codebook builder tool
            res_build = tools["bow_build"].run(root, ImageDocument())
            self.assertIsNotNone(res_build)
            self.assertIn("BoW Codebook Construction", res_build.details["Operation"])

            # Histogram tool on single image with top 5 words
            res_hist = tools["bow_histogram"].run(root, self.document)
            self.assertIsNotNone(res_hist)
            self.assertIsNone(res_hist.image, "Must not replace current working image")
            self.assertIn("Rank #1", res_hist.details)
            self.assertIn("Rank #5", res_hist.details)

            # Localize window test
            win = VisualWordLocalizationWindow(
                root,
                self.image,
                "test_img",
                load_kmeans(),
                load_visual_words(),
                initial_word=1,
            )
            self.assertIsNotNone(win)
            win.destroy()

            # Retrieval tool
            res_ret = tools["bow_retrieval"].run(root, self.document)
            self.assertIsNotNone(res_ret)
            self.assertIsNone(res_ret.image, "Must not replace current working image")
            self.assertIn("Content-Based Image Retrieval", res_ret.details["Operation"])
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
