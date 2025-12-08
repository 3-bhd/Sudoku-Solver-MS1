import numpy as np
import cv2

def create_multiple_templates_per_digit():
    templates = {}

    for digit in range(1, 10):
        digit_templates = []

        for font, scale in [
            (cv2.FONT_HERSHEY_SIMPLEX, 1.0),
            (cv2.FONT_HERSHEY_DUPLEX, 0.9)
        ]:
            for thickness in [1, 2, 3]:
                img = np.zeros((32, 32), dtype=np.uint8)
                size = cv2.getTextSize(str(digit), font, scale, thickness)[0]
                text_x = (32 - size[0]) // 2
                text_y = (32 + size[1]) // 2

                cv2.putText(img, str(digit), (text_x, text_y),
                            font, scale, 255, thickness, cv2.LINE_AA)
                digit_templates.append(img)

        templates[digit] = digit_templates

    return templates
