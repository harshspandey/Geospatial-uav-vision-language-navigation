import torch
import clip
import numpy as np
from PIL import Image
import cv2
import math


class VSV:
    """
    Visual Sub-Goal Verifier.
    Uses frozen CLIP ViT-L/14 to check if a landmark is visually
    present in the UAV's current view.
    Combines visual similarity + distance for robust sub-goal completion.
    """

    def __init__(self, threshold=0.70, distance_threshold=40.0, device=None):
        self.threshold = threshold
        self.distance_threshold = distance_threshold
        self.device = device or "cpu"

        print(f"[VSV] Loading CLIP ViT-L/14 on {self.device}...")
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.model.eval()
        print("[VSV] CLIP loaded.")

        # Cache for reference embeddings
        self._ref_cache = {}

    def _encode_image(self, image_path):
        """Encode an image file to CLIP embedding."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy()

    def _encode_image_array(self, image_array):
        """Encode a numpy image array to CLIP embedding."""
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        image = Image.fromarray(image_array).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy()

    def _encode_text(self, text):
        """Encode a text description to CLIP embedding."""
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy()

    def _cosine_similarity(self, a, b):
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(a.flatten(), b.flatten()))

    def get_reference_embedding(self, map_image_path, landmark_bbox, cache_key=None):
        """
        Extract reference embedding from landmark bbox crop on the map.
        Args:
            map_image_path: path to the full semantic map image
            landmark_bbox: [x1, y1, x2, y2] pixel coordinates
            cache_key: optional key for caching
        """
        if cache_key and cache_key in self._ref_cache:
            return self._ref_cache[cache_key]

        try:
            image = cv2.imread(map_image_path)
            if image is None:
                return None

            x1, y1, x2, y2 = landmark_bbox
            h, w = image.shape[:2]

            # Clamp to image bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            crop = image[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            embedding = self._encode_image_array(crop_rgb)

            if cache_key:
                self._ref_cache[cache_key] = embedding

            return embedding

        except Exception as e:
            print(f"[VSV] Reference embedding error: {e}")
            return None

    def verify(self, 
               drone_view_path,
               subgoal_description,
               map_image_path=None,
               landmark_bbox=None,
               current_pos_world=None,
               subgoal_waypoint=None,
               px_real_size=None):
        """
        Check if sub-goal is visually confirmed.
        
        Args:
            drone_view_path: path to UAV first-person RGB image
            subgoal_description: text description of the sub-goal landmark
            map_image_path: path to semantic map (for bbox crop reference)
            landmark_bbox: [x1,y1,x2,y2] of landmark on map
            current_pos_world: current UAV world position (x, y)
            subgoal_waypoint: sub-goal waypoint in pixel coords [px, py]
            px_real_size: [meters_per_pixel_x, meters_per_pixel_y]
        
        Returns:
            (confirmed: bool, similarity: float, distance: float)
        """
        similarity = 0.0
        distance = float('inf')

        # --- Visual similarity check ---
        try:
            # Method 1: text-image similarity (always available)
            query_emb = self._encode_image(drone_view_path)
            text_emb = self._encode_text(f"aerial view of {subgoal_description}")
            similarity = self._cosine_similarity(query_emb, text_emb)

            # Method 2: image-image similarity (if map crop available)
            if map_image_path and landmark_bbox:
                cache_key = f"{map_image_path}_{landmark_bbox}"
                ref_emb = self.get_reference_embedding(
                    map_image_path, landmark_bbox, cache_key
                )
                if ref_emb is not None:
                    img_similarity = self._cosine_similarity(query_emb, ref_emb)
                    # Take max of text and image similarity
                    similarity = max(similarity, img_similarity)

        except Exception as e:
            print(f"[VSV] Similarity error: {e}")
            return False, 0.0, float('inf')

        # --- Distance check (if world coords available) ---
        distance_ok = True
        if current_pos_world and subgoal_waypoint and px_real_size:
            try:
                # Convert waypoint pixels to world coords (rough estimate)
                wp_world_x = subgoal_waypoint[0] * px_real_size[0]
                wp_world_y = subgoal_waypoint[1] * px_real_size[1]
                dist_x = current_pos_world[0] - wp_world_x
                dist_y = current_pos_world[1] - wp_world_y
                distance = math.sqrt(dist_x**2 + dist_y**2)
                distance_ok = distance < self.distance_threshold
            except Exception:
                distance_ok = True  # if distance fails, rely on visual only

        # --- Combined decision ---
        visual_ok = similarity > self.threshold
        confirmed = visual_ok and distance_ok

        return confirmed, similarity, distance

    def calibrate_threshold(self, val_seen_results):
        """
        Calibrate threshold from validation results.
        val_seen_results: list of (similarity, is_correct_completion)
        """
        if not val_seen_results:
            return self.threshold

        correct_sims = [s for s, correct in val_seen_results if correct]
        wrong_sims = [s for s, correct in val_seen_results if not correct]

        if correct_sims and wrong_sims:
            # Pick threshold at midpoint between distributions
            self.threshold = (np.mean(correct_sims) + np.mean(wrong_sims)) / 2
            print(f"[VSV] Calibrated threshold: {self.threshold:.3f}")
            print(f"[VSV] Correct sim mean: {np.mean(correct_sims):.3f}")
            print(f"[VSV] Wrong sim mean: {np.mean(wrong_sims):.3f}")

        return self.threshold
