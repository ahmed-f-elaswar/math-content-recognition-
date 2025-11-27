"""Bounding box manipulation and processing utilities.

This module provides functions for working with bounding boxes in images:
- Masking regions
- Merging overlapping boxes
- Resolving conflicts between text and formula regions
- Slicing image regions
- Drawing boxes for visualization
"""

import heapq
import os
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image, ImageDraw
from texteller.types import Bbox

_MAXV = 999999999  # Large value used as sentinel for comparison


def mask_img(img: np.ndarray, bboxes: List[Bbox], bg_color: np.ndarray) -> np.ndarray:
	"""Mask specified regions in an image with a background color.
	
	This is useful for hiding detected formula regions before performing
	text OCR, preventing the OCR from trying to recognize formulas as text.
	
	Args:
		img: Input image as numpy array (H, W, C)
		bboxes: List of bounding boxes to mask
		bg_color: Background color to fill masked regions (RGB array)
		
	Returns:
		Image with specified regions masked
	"""
	mask_img = img.copy()
	for bbox in bboxes:
		mask_img[bbox.p.y : bbox.p.y + bbox.h, bbox.p.x : bbox.p.x + bbox.w] = bg_color
	return mask_img


def bbox_merge(sorted_bboxes: List[Bbox]) -> List[Bbox]:
	"""Merge horizontally adjacent or overlapping bounding boxes on the same row.
	
	This function combines bounding boxes that are on the same horizontal line
	and either overlap or are very close to each other. This is useful for
	combining detected regions that belong to the same text or formula.
	
	Args:
		sorted_bboxes: List of bounding boxes sorted by position
		              (use Bbox.__lt__ for proper sorting)
		
	Returns:
		List of merged bounding boxes
		
	Example:
		>>> boxes = [Bbox(0, 0, 10, 10), Bbox(8, 0, 10, 10)]  # Overlapping
		>>> merged = bbox_merge(boxes)
		>>> len(merged)
		1
	"""
	if len(sorted_bboxes) == 0:
		return []
	bboxes = sorted_bboxes.copy()
	guard = Bbox(_MAXV, bboxes[-1].p.y, -1, -1, label="guard")
	bboxes.append(guard)
	res = []
	prev = bboxes[0]
	for curr in bboxes:
		if prev.ur_point.x <= curr.p.x or not prev.same_row(curr):
			res.append(prev)
			prev = curr
		else:
			prev.w = max(prev.w, curr.ur_point.x - prev.p.x)
	return res


def split_conflict(ocr_bboxes: List[Bbox], latex_bboxes: List[Bbox]) -> List[Bbox]:
	"""Resolve conflicts between OCR text boxes and LaTeX formula boxes.
	
	When text detection and formula detection produce overlapping regions,
	this function intelligently splits and adjusts the text boxes to avoid
	overlapping with formula regions. LaTeX boxes take priority.
	
	The algorithm:
	1. Sorts all boxes by position
	2. For overlapping regions, gives priority to formula boxes
	3. Splits or trims text boxes to not overlap with formulas
	4. Preserves all formula boxes intact
	
	Args:
		ocr_bboxes: List of text bounding boxes from OCR
		latex_bboxes: List of formula bounding boxes
		
	Returns:
		List of text boxes with conflicts resolved (no overlap with formulas)
	"""
	if latex_bboxes == []:
		return ocr_bboxes
	if ocr_bboxes == [] or len(ocr_bboxes) == 1:
		return ocr_bboxes

	bboxes = sorted(ocr_bboxes + latex_bboxes)

	assert len(bboxes) > 1

	heapq.heapify(bboxes)
	res = []
	candidate = heapq.heappop(bboxes)
	curr = heapq.heappop(bboxes)
	idx = 0
	while len(bboxes) > 0:
		idx += 1
		assert candidate.p.x <= curr.p.x or not candidate.same_row(curr)

		if candidate.ur_point.x <= curr.p.x or not candidate.same_row(curr):
			res.append(candidate)
			candidate = curr
			curr = heapq.heappop(bboxes)
		elif candidate.ur_point.x < curr.ur_point.x:
			assert not (candidate.label != "text" and curr.label != "text")
			if candidate.label == "text" and curr.label == "text":
				candidate.w = curr.ur_point.x - candidate.p.x
				curr = heapq.heappop(bboxes)
			elif candidate.label != curr.label:
				if candidate.label == "text":
					candidate.w = curr.p.x - candidate.p.x
					res.append(candidate)
					candidate = curr
					curr = heapq.heappop(bboxes)
				else:
					curr.w = curr.ur_point.x - candidate.ur_point.x
					curr.p.x = candidate.ur_point.x
					heapq.heappush(bboxes, curr)
					curr = heapq.heappop(bboxes)

		elif candidate.ur_point.x >= curr.ur_point.x:
			assert not (candidate.label != "text" and curr.label != "text")

			if candidate.label == "text":
				assert curr.label != "text"
				heapq.heappush(
					bboxes,
					Bbox(
						curr.ur_point.x,
						candidate.p.y,
						candidate.h,
						candidate.ur_point.x - curr.ur_point.x,
						label="text",
						confidence=candidate.confidence,
						content=None,
					),
				)
				candidate.w = curr.p.x - candidate.p.x
				res.append(candidate)
				candidate = curr
				curr = heapq.heappop(bboxes)
			else:
				assert curr.label == "text"
				curr = heapq.heappop(bboxes)
		else:
			assert False
	res.append(candidate)
	res.append(curr)

	return res


def slice_from_image(img: np.ndarray, ocr_bboxes: List[Bbox]) -> List[np.ndarray]:
	"""Extract image regions defined by bounding boxes.
	
	Slices out rectangular regions from an image based on bounding boxes.
	Useful for extracting detected text or formula regions for recognition.
	
	Args:
		img: Source image as numpy array (H, W, C)
		ocr_bboxes: List of bounding boxes defining regions to extract
		
	Returns:
		List of image patches (numpy arrays) corresponding to each bbox
	"""
	sliced_imgs = []
	for bbox in ocr_bboxes:
		x, y = int(bbox.p.x), int(bbox.p.y)
		w, h = int(bbox.w), int(bbox.h)
		sliced_img = img[y : y + h, x : x + w]
		sliced_imgs.append(sliced_img)
	return sliced_imgs


def draw_bboxes(img: Image.Image, bboxes: List[Bbox], name: str = "annotated_image.png") -> None:
	"""Draw bounding boxes on an image for visualization and debugging.
	
	Draws rectangles around detected regions and labels them with their
	type and content (if available). Saves the annotated image to a logs folder.
	
	Args:
		img: PIL Image to draw on
		bboxes: List of bounding boxes to draw
		name: Filename to save the annotated image (default: 'annotated_image.png')
		
	Example:
		>>> from PIL import Image
		>>> img = Image.open('document.png')
		>>> boxes = latex_detect('document.png', model)
		>>> draw_bboxes(img, boxes, 'debug_detection.png')
	"""
	curr_work_dir = Path(os.getcwd())
	log_dir = curr_work_dir / "logs"
	log_dir.mkdir(exist_ok=True)
	drawer = ImageDraw.Draw(img)
	for bbox in bboxes:
		# Calculate the coordinates for the rectangle to be drawn
		left = bbox.p.x
		top = bbox.p.y
		right = bbox.p.x + bbox.w
		bottom = bbox.p.y + bbox.h

		# Draw the rectangle on the image
		drawer.rectangle([left, top, right, bottom], outline="green", width=1)

		# Optionally, add text label if it exists
		if bbox.label:
			drawer.text((left, top), bbox.label, fill="blue")

		if bbox.content:
			drawer.text((left, bottom - 10), bbox.content[:10], fill="red")

	# Save the image with drawn rectangles
	img.save(log_dir / name)
