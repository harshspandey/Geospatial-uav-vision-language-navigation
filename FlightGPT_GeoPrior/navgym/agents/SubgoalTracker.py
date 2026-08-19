import re
import json


class SubgoalTracker:
    """
    Tracks hierarchical sub-goals between model calls.
    Maintains state: which sub-goals are done, which is active, 
    and triggers fallback if stuck too long.
    """

    def __init__(self, stuck_limit=5):
        self.subgoals = []          # list of {id, description, waypoint, landmark_bbox}
        self.current_idx = 0        # index of active sub-goal
        self.stuck_counter = 0      # steps stuck on same sub-goal
        self.stuck_limit = stuck_limit
        self.completion_log = []    # log of completed sub-goals

    def update_from_model_output(self, model_output):
        """Parse <global_plan> and <current_subgoal> from model output."""
        # Parse global plan steps
        plan_match = re.search(
            r'<global_plan>(.*?)</global_plan>', 
            model_output, re.DOTALL
        )
        if plan_match:
            plan_text = plan_match.group(1).strip()
            steps = re.findall(
                r'Step\s+(\d+):\s+(.+?)\s*->\s*waypoint:\s*\[(\d+),\s*(\d+)\]',
                plan_text
            )
            if steps:
                self.subgoals = []
                for step_id, desc, wp_x, wp_y in steps:
                    self.subgoals.append({
                        'id': int(step_id),
                        'description': desc.strip(),
                        'waypoint': [int(wp_x), int(wp_y)],
                        'landmark_bbox': None
                    })

        # Parse current subgoal bbox
        subgoal_match = re.search(
            r'<current_subgoal>(.*?)</current_subgoal>',
            model_output, re.DOTALL
        )
        if subgoal_match:
            try:
                subgoal_text = subgoal_match.group(1).strip()
                subgoal_data = json.loads(subgoal_text)
                if self.subgoals and 'landmark_bbox' in subgoal_data:
                    self.subgoals[0]['landmark_bbox'] = subgoal_data['landmark_bbox']
            except Exception:
                pass

    def get_active_subgoal(self):
        """Return the currently active sub-goal or None if all done."""
        if self.current_idx < len(self.subgoals):
            return self.subgoals[self.current_idx]
        return None

    def advance(self):
        """Mark current sub-goal complete and move to next."""
        if self.current_idx < len(self.subgoals):
            completed = self.subgoals[self.current_idx]
            self.completion_log.append(completed)
            self.current_idx += 1
            self.stuck_counter = 0
            return True
        return False

    def increment_stuck(self):
        """Call each step when sub-goal not yet completed."""
        self.stuck_counter += 1

    def is_stuck(self):
        """Return True if stuck on same sub-goal too long."""
        return self.stuck_counter >= self.stuck_limit

    def reset_stuck(self):
        """Reset stuck counter after fallback."""
        self.stuck_counter = 0

    def get_completion_summary(self):
        """Return summary for metrics logging."""
        return {
            'total_subgoals': len(self.subgoals),
            'completed_subgoals': len(self.completion_log),
            'completion_rate': len(self.completion_log) / max(len(self.subgoals), 1),
            'completed_descriptions': [s['description'] for s in self.completion_log]
        }

    def get_context_for_prompt(self):
        """Return a string to inject into the model prompt about sub-goal progress."""
        if not self.subgoals:
            return ""
        
        completed = self.completion_log
        active = self.get_active_subgoal()
        
        lines = []
        if completed:
            lines.append(f"[Completed sub-goals: {', '.join([s['description'] for s in completed])}]")
        if active:
            lines.append(f"[Current sub-goal {active['id']}: {active['description']} → waypoint {active['waypoint']}]")
        
        return "\n".join(lines)

    def has_subgoals(self):
        """Return True if model output contained a valid global plan."""
        return len(self.subgoals) > 0

    def all_complete(self):
        """Return True if all sub-goals are done."""
        return self.current_idx >= len(self.subgoals) and len(self.subgoals) > 0
