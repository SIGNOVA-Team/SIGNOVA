"""Utilities for temporal hand-slot tracking."""

import numpy as np


def track_hand_slots(keypoints, max_distance=0.15, max_gap=15):
    """Match detected hands to slots using recent wrist positions."""
    hand_count = keypoints.shape[1]
    tracked = np.full_like(keypoints, np.nan)
    last_positions = np.full((hand_count, 2), np.nan)
    last_seen = np.full(hand_count, -1, dtype=int)
    for frame_index, frame in enumerate(keypoints):
        detected_slots = np.flatnonzero(np.isfinite(frame[:, :2]).all(axis=1))
        detections = frame[detected_slots]
        if len(detections) == 0:
            continue
        candidates = []
        for detection_index, detection in enumerate(detections):
            for slot in range(hand_count):
                gap = frame_index - last_seen[slot]
                if not 0 < gap <= max_gap:
                    continue
                distance = np.linalg.norm(detection[:2] - last_positions[slot])
                if distance <= max_distance:
                    candidates.append((distance, detection_index, slot))
        matched_detections = set()
        matched_slots = set()
        for _, detection_index, slot in sorted(candidates):
            if detection_index in matched_detections or slot in matched_slots:
                continue
            tracked[frame_index, slot] = detections[detection_index]
            matched_detections.add(detection_index)
            matched_slots.add(slot)
        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            free_slots = [slot for slot in range(hand_count) if slot not in matched_slots]
            preferred_slot = int(detected_slots[detection_index])
            if preferred_slot in free_slots:
                assigned_slot = preferred_slot
            else:
                assigned_slot = free_slots[0]
            tracked[frame_index, assigned_slot] = detection
            matched_slots.add(assigned_slot)
        for slot in matched_slots:
            last_positions[slot] = tracked[frame_index, slot, :2]
            last_seen[slot] = frame_index
    return tracked
