import cv2
import numpy as np
from cell_processing import is_cell_empty


def extract_digit_from_cell(cell):
    """
    Extracts the digit from a cell by finding the largest contour
    and cropping/centering it.
    IMPROVED: Handles very thin digits like 1 with better detection logic
    """
    if is_cell_empty(cell):
        return None

    # Find contours
    contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Find the largest contour (should be the digit)
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)

    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate cell dimensions for relative comparisons
    cell_height, cell_width = cell.shape
    cell_area = cell_height * cell_width

    # More lenient filtering for noise
    # Key fix: Use area ratio instead of just pixel dimensions
    area_ratio = contour_area / cell_area

    # Reject contours that are too small (noise)
    # Allow very thin digits like "1" by focusing on area ratio
    if area_ratio < 0.01:  # 1% of cell area
        return None

    # Digits should be reasonably constrained within the cell
    # Reject if bounding box covers almost the entire cell (probably noise)
    if contour_area > 0.9 * cell_area:
        return None

    # Minimum height threshold (relative)
    # We focus on height because digits are tall even if thin
    if h < 0.2 * cell_height:  # At least 20% of the cell height
        return None

    # Aspect ratio: allow very tall digits (thin "1"), but reject extremely flat noise
    aspect_ratio = h / float(w)

    # Digits should be taller than wide (aspect > 1.0)
    # But can be VERY tall (up to 10:1 for thin "1"s)
    if aspect_ratio < 1.0 or aspect_ratio > 10.0:
        return None

    # Additional check: digit should be reasonably centered
    # This helps reject edge noise
    x_center = x + w / 2
    y_center = y + h / 2
    cell_x_center = cell_width / 2
    cell_y_center = cell_height / 2

    # Allow digit to be within 40% of cell from center
    max_offset_x = cell_width * 0.4
    max_offset_y = cell_height * 0.4

    if abs(x_center - cell_x_center) > max_offset_x or abs(y_center - cell_y_center) > max_offset_y:
        return None

    # Add padding around the digit to avoid cutting off parts
    pad_y = int(0.10 * h)  # 10% of height
    pad_x = int(0.10 * w)  # 10% of width

    y1 = max(y - pad_y, 0)
    y2 = min(y + h + pad_y, cell_height)
    x1 = max(x - pad_x, 0)
    x2 = min(x + w + pad_x, cell_width)

    digit = cell[y1:y2, x1:x2]

    # Make sure we have a non-empty ROI
    if digit.size == 0:
        return None

    # Normalize the digit image size
    target_size = 28  # Target size (28x28)

    digit_h, digit_w = digit.shape

    # If the digit is much taller than wide (thin "1"), we need extra horizontal padding
    if digit_h > digit_w:
        # First, ensure the height matches the target height (for all digits, especially 1)
        # Calculate padding needed to make square
        diff = digit_h - digit_w
        pad_left = diff // 2
        pad_right = diff - pad_left

        # For very thin digits like 1, add extra padding
        if digit_w < digit_h * 0.3:  # Very thin digit
            extra_pad = int(digit_h * 0.15)  # Add 15% extra
            pad_left += extra_pad
            pad_right += extra_pad

        digit = cv2.copyMakeBorder(digit, 0, 0, pad_left, pad_right,
                                   cv2.BORDER_CONSTANT, value=0)
    else:
        # Width is larger (rare for normal digits)
        diff = digit_w - digit_h
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        digit = cv2.copyMakeBorder(digit, pad_top, pad_bottom, 0, 0,
                                   cv2.BORDER_CONSTANT, value=0)

    # Resize to target size
    digit_resized = cv2.resize(digit, (target_size, target_size),
                               interpolation=cv2.INTER_AREA)

    # Add small border for better matching
    digit_final = cv2.copyMakeBorder(digit_resized, 2, 2, 2, 2,
                                     cv2.BORDER_CONSTANT, value=0)

    return digit_final


def match_digit_with_contour_analysis(digit_img, templates, img_name=None, cell_pos=None):
    """
    Enhanced digit matching using contour shape analysis + template matching.
    Now includes closed/open digit detection for better 6/8/9/0 differentiation.
    Always returns the best match (never rejects based on gap).
    """
    if digit_img is None:
        return 0, 0.0, {}

    # Normalize the digit
    digit_normalized = cv2.normalize(digit_img, None, 0, 255, cv2.NORM_MINMAX)

    # Prepare variations of the input digit
    kernel = np.ones((2, 2), np.uint8)
    digit_thinned = cv2.erode(digit_normalized, kernel, iterations=1)
    digit_thickened = cv2.dilate(digit_normalized, kernel, iterations=1)

    # Collect all variations
    digit_variants = {
        "original": digit_normalized,
        "thinned": digit_thinned,
        "thickened": digit_thickened
    }

    # Precompute contours and hierarchy for the original digit
    contours, hierarchy = cv2.findContours(digit_normalized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize contour features
    num_holes = 0
    input_contour_features = {
        'area_ratio': 0,
        'perimeter': 0,
        'aspect_ratio': 0,
        'extent': 0,
        'solidity': 0,
        'hu_moments': np.zeros(3)
    }

    if contours and hierarchy is not None:
        # Find main contour (largest area)
        main_contour = max(contours, key=cv2.contourArea)

        # Count holes (children in contour hierarchy)
        holes = 0
        for idx, h_info in enumerate(hierarchy[0]):
            parent_idx = h_info[3]
            if parent_idx != -1:  # Has a parent → this contour is a hole
                area = cv2.contourArea(contours[idx])
                if area > 5:  # Ignore tiny holes/noise
                    holes += 1
        num_holes = holes

        # Feature 1: Contour area ratio
        input_contour_features['area_ratio'] = cv2.contourArea(main_contour) / (
                    digit_normalized.shape[0] * digit_normalized.shape[1])

        # Feature 2: Perimeter
        input_contour_features['perimeter'] = cv2.arcLength(main_contour, True)

        # Feature 3: Bounding box aspect ratio
        x, y, w, h = cv2.boundingRect(main_contour)
        input_contour_features['aspect_ratio'] = h / w if w > 0 else 0

        # Feature 4: Extent (ratio of contour area to bounding box area)
        input_contour_features['extent'] = cv2.contourArea(main_contour) / (w * h) if (w * h) > 0 else 0

        # Feature 5: Solidity (ratio of contour area to convex hull area)
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        input_contour_features['solidity'] = cv2.contourArea(main_contour) / hull_area if hull_area > 0 else 0

        # Feature 6: Hu Moments
        moments = cv2.moments(main_contour)
        hu = cv2.HuMoments(moments).flatten()
        # Take the log transform (for better numerical stability)
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        input_contour_features['hu_moments'] = hu[:3]  # Just first 3 for simplicity

    # Prepare a dict to store scores for each digit
    digit_scores = {}
    debug_info = {
        'template_scores': {},
        'contour_features': input_contour_features,
        'num_holes': num_holes
    }

    # Template matching + contour-based analysis for each digit
    for digit, digit_templates in templates.items():
        best_score = -1
        scores = []

        # Compare input digit against all templates of this digit
        for digit_variant_name, digit_variant in digit_variants.items():
            for template_idx, template in enumerate(digit_templates):
                # Ensure template and digit_variant have same size
                if digit_variant.shape != template.shape:
                    resized_variant = cv2.resize(digit_variant, (template.shape[1], template.shape[0]))
                else:
                    resized_variant = digit_variant

                # Compute template matching scores
                result_ccoeff = cv2.matchTemplate(resized_variant, template, cv2.TM_CCOEFF_NORMED)
                result_ccorr = cv2.matchTemplate(resized_variant, template, cv2.TM_CCORR_NORMED)
                result_sqdiff = cv2.matchTemplate(resized_variant, template, cv2.TM_SQDIFF_NORMED)

                score_ccoeff = float(result_ccoeff.max())
                score_ccorr = float(result_ccorr.max())
                score_sqdiff = 1.0 - float(result_sqdiff.min())  # Invert SQDIFF (lower is better)

                # Combined score using weighted sum of metrics
                combined_score = (
                        0.65 * score_ccoeff +
                        0.20 * score_ccorr +
                        0.15 * score_sqdiff
                )

                scores.append({
                    'variant': digit_variant_name,
                    'template_index': template_idx,
                    'score_ccoeff': score_ccoeff,
                    'score_ccorr': score_ccorr,
                    'score_sqdiff': score_sqdiff,
                    'combined_score': combined_score
                })

                if combined_score > best_score:
                    best_score = combined_score

        # Store the best score and detailed scores for debugging
        digit_scores[digit] = {
            'combined': best_score,
            'details': scores
        }

    # Store template scores for debugging
    debug_info['template_scores'] = digit_scores

    # Use contour features and hole count to adjust scores (for digits like 6, 8, 9)
    # Define expected behavior based on number of holes and contour properties
    for digit, score_info in digit_scores.items():
        hole_weight = 0.0
        aspect_weight = 0.0

        # Digits that typically have no holes: 1, 2, 3, 5, 7
        if digit in [1, 2, 3, 5, 7]:
            if num_holes == 0:
                hole_weight = 0.02
            elif num_holes >= 1:
                hole_weight = -0.15

        # Digits that should have exactly one hole: 6, 9
        elif digit in [6, 9]:
            if num_holes == 1:
                hole_weight = 0.05
            elif num_holes == 0:
                hole_weight = -0.10
            else:
                hole_weight = -0.20

        # Digits that often have two holes: 8
        elif digit == 8:
            if num_holes >= 2:
                hole_weight = 0.08
            elif num_holes == 1:
                hole_weight = -0.08
            else:
                hole_weight = -0.15

        # Digit 0 (if you supported it) would have 1 hole:
        # left here conceptually, but not in [1..9] range.

        # Aspect ratio adjustment (1 is very tall & thin, 2/3/5 maybe slightly tall, 8/0 more round)
        aspect_ratio = input_contour_features.get('aspect_ratio', 1.0)
        if digit == 1:
            # Encourage tall, thin lines
            if aspect_ratio > 4.0:
                aspect_weight = 0.05
            elif aspect_ratio < 2.0:
                aspect_weight = -0.05
        elif digit == 8:
            # Encourage more "roundish" shapes
            if 1.0 < aspect_ratio < 2.5:
                aspect_weight = 0.03
            elif aspect_ratio > 3.5:
                aspect_weight = -0.05

        # Hu moments-based rule could be added here too if needed

        # Adjust final score for the digit
        adjusted_score = score_info['combined'] + hole_weight + aspect_weight
        digit_scores[digit]['combined'] = adjusted_score

    # Choose the best digit based on adjusted combined scores
    best_digit = max(digit_scores, key=lambda d: digit_scores[d]['combined'])
    best_score = digit_scores[best_digit]['combined']

    # Hole-informed correction for typical confusions
    # ----------------------------------------------------
    # Specific rules for 6/8/9 confusion based on hole count
    if best_digit in [6, 8, 9]:
        # Scores for these digits
        score_6 = digit_scores[6]['combined']
        score_8 = digit_scores[8]['combined']
        score_9 = digit_scores[9]['combined']

        # Rule 1: If we detect 6 but see 2 strong holes, likely 8
        if best_digit == 6 and num_holes >= 2:
            if score_8 > 0.35 and (score_6 - score_8) < 0.1:
                return 8, score_8, debug_info

        # Rule 2: If we detect 8 but see only 1 hole, it may be 6 or 9
        if best_digit == 8 and num_holes == 1:
            # If 6 is close in score, prefer 6 because 6's loop is often smaller
            if (score_6 > 0.35 and (score_8 - score_6) < 0.10 and score_6 >= score_9):
                return 6, score_6, debug_info
            # Or if 9 is close
            if (score_9 > 0.35 and (score_8 - score_9) < 0.10 and score_9 >= score_6):
                return 9, score_9, debug_info

        # Rule 3: If we detect 9 but digit has 2 holes, it's likely 8
        if best_digit == 9 and num_holes >= 2:
            if score_8 > 0.35 and (score_9 - score_8) < 0.10:
                return 8, score_8, debug_info

    # Additional rule for 0 vs 8 confusion (if 0 were supported)
    # - Not used here since our digit range is 1-9

    # Tightening 2 vs 1 distinction:
    # If best digit is 2, but shape is extremely thin and tall (like 1), we reconsider:
    if best_digit == 2:
        aspect_ratio = input_contour_features.get('aspect_ratio', 1.0)
        area_ratio = input_contour_features.get('area_ratio', 0.0)
        # Extremely tall/skinny shape with small area may be 1
        if aspect_ratio > 5.0 and area_ratio < 0.10:
            digit_scores[1]['combined'] += 0.05  # small boost to 1
            # Re-evaluate best digit
            best_digit = max(digit_scores, key=lambda d: digit_scores[d]['combined'])
            best_score = digit_scores[best_digit]['combined']

    # 4 vs 9 confusion rule:
    # If best digit is 9, but aspect ratio is small and shape is more open,
    # consider 4 as an alternative if its score is close.
    if best_digit == 9:
        aspect_ratio = input_contour_features.get('aspect_ratio', 1.0)
        # "4" is usually fairly tall, but top is often open / not fully looped
        if aspect_ratio > 2.0 and num_holes == 0:
            # If 4 is close enough in score, choose 4
            if digit_scores[4]['combined'] > 0.30 and \
                    (digit_scores[9]['combined'] - digit_scores[4]['combined']) < 0.07:
                return 4, digit_scores[4]['combined'], debug_info

    # Always return the best digit (no absolute threshold)
    return best_digit, best_score, debug_info
