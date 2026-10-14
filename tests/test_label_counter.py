import unittest

import cv2
import numpy as np

from label_counter import count_white_labels


def encode(image):
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("测试图片编码失败")
    return data.tobytes()


class WhiteLabelCounterTest(unittest.TestCase):
    def test_counts_separate_white_rectangular_labels(self):
        image = np.full((700, 1000, 3), (55, 80, 105), dtype=np.uint8)
        expected = [(80, 70, 220, 130), (420, 110, 250, 150), (180, 390, 280, 165)]
        for x, y, width, height in expected:
            cv2.rectangle(image, (x, y), (x + width, y + height), (245, 245, 245), -1)
            cv2.rectangle(image, (x + 22, y + 24), (x + width - 22, y + 31), (40, 40, 40), -1)
            cv2.rectangle(image, (x + 22, y + 48), (x + width - 60, y + 55), (70, 70, 70), -1)

        result = count_white_labels(encode(image))

        self.assertEqual(3, result["count"])
        self.assertEqual(3, len(result["boxes"]))
        self.assertEqual({"width": 1000, "height": 700}, result["image"])

    def test_ignores_small_white_noise_and_full_white_frame(self):
        image = np.full((600, 800, 3), (45, 70, 90), dtype=np.uint8)
        for x in range(20, 780, 60):
            cv2.rectangle(image, (x, 20), (x + 8, 28), (255, 255, 255), -1)
        self.assertEqual(0, count_white_labels(encode(image))["count"])

        white_frame = np.full((600, 800, 3), 255, dtype=np.uint8)
        self.assertEqual(0, count_white_labels(encode(white_frame))["count"])

    def test_rejects_invalid_image(self):
        with self.assertRaisesRegex(ValueError, "无法解码"):
            count_white_labels(b"not-an-image")


if __name__ == "__main__":
    unittest.main()
