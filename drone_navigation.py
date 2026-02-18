"""
Drone Navigation System - Phase 2
Analyzes obstacle positions and suggests drone movement directions
"""

import cv2
import numpy as np
from collections import Counter
import math

class DroneNavigator:
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2
        
        # Define navigation zones
        self.zones = {
            'center': (frame_width//3, frame_height//3, 2*frame_width//3, 2*frame_height//3),
            'left': (0, 0, frame_width//3, frame_height),
            'right': (2*frame_width//3, 0, frame_width, frame_height),
            'top': (0, 0, frame_width, frame_height//3),
            'bottom': (0, 2*frame_height//3, frame_width, frame_height)
        }
    
    def analyze_obstacles(self, detections):
        """
        Analyze obstacle positions and determine safe navigation direction
        
        Args:
            detections: List of [x1, y1, x2, y2, conf, class] detections
            
        Returns:
            dict: Navigation recommendations
        """
        if not detections or len(detections) == 0:
            return {
                'status': 'CLEAR_PATH',
                'direction': 'FORWARD',
                'confidence': 1.0,
                'obstacles_by_zone': {},
                'recommendation': 'Path is clear - proceed forward'
            }
        
        # Count obstacles in each zone
        zone_obstacles = {zone: [] for zone in self.zones.keys()}
        
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Check which zones this obstacle occupies
            for zone_name, (zx1, zy1, zx2, zy2) in self.zones.items():
                if zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2:
                    zone_obstacles[zone_name].append({
                        'center': (center_x, center_y),
                        'bbox': (x1, y1, x2, y2),
                        'size': (x2-x1) * (y2-y1),
                        'class': int(det[5]) if len(det) > 5 else 0
                    })
        
        # Determine navigation strategy
        return self._calculate_navigation(zone_obstacles)
    
    def _calculate_navigation(self, zone_obstacles):
        """Calculate best navigation direction based on obstacle distribution"""
        
        # Count obstacles per zone
        zone_counts = {zone: len(obs) for zone, obs in zone_obstacles.items()}
        
        # Calculate danger scores (more obstacles = higher danger)
        danger_scores = {}
        for zone, obstacles in zone_obstacles.items():
            if not obstacles:
                danger_scores[zone] = 0
            else:
                # Factor in number, size, and proximity of obstacles
                total_size = sum(obs['size'] for obs in obstacles)
                # Add proximity factor - obstacles closer to center are more dangerous
                proximity_penalty = 0
                for obs in obstacles:
                    cx, cy = obs['center']
                    dist_from_center = abs(cx - self.center_x) + abs(cy - self.center_y)
                    proximity_penalty += (1.0 - (dist_from_center / (self.frame_width + self.frame_height))) * 0.5
                
                danger_scores[zone] = len(obstacles) * 0.5 + (total_size / 10000) * 0.3 + proximity_penalty * 0.2
        
        # Determine navigation recommendation
        center_danger = danger_scores['center']
        
        if center_danger == 0:
            return {
                'status': 'CLEAR_PATH',
                'direction': 'FORWARD',
                'confidence': 1.0,
                'obstacles_by_zone': zone_counts,
                'recommendation': 'Center path clear - proceed forward'
            }
        
        # Find safest direction with better tie-breaking
        side_dangers = {
            'LEFT': danger_scores['left'],
            'RIGHT': danger_scores['right'],
            'UP': danger_scores['top'],
            'DOWN': danger_scores['bottom']
        }
        
        # Sort by danger score to handle ties better
        sorted_directions = sorted(side_dangers.items(), key=lambda x: (x[1], x[0]))
        safest_direction = sorted_directions[0][0]
        safest_score = sorted_directions[0][1]
        
        # If left and right have similar scores, prefer horizontal movement over vertical
        if abs(side_dangers['LEFT'] - side_dangers['RIGHT']) < 0.1:
            # Both sides are similar, choose the one with fewer obstacles
            if zone_counts.get('left', 0) < zone_counts.get('right', 0):
                safest_direction = 'LEFT'
                safest_score = side_dangers['LEFT']
            elif zone_counts.get('left', 0) > zone_counts.get('right', 0):
                safest_direction = 'RIGHT'
                safest_score = side_dangers['RIGHT']
            # If equal, check which has more free space
            elif zone_counts.get('left', 0) == zone_counts.get('right', 0):
                # Calculate average position of center obstacles
                if zone_obstacles['center']:
                    avg_x = sum(obs['center'][0] for obs in zone_obstacles['center']) / len(zone_obstacles['center'])
                    # If obstacles are more to the left, go right
                    if avg_x < self.center_x:
                        safest_direction = 'RIGHT'
                        safest_score = side_dangers['RIGHT']
                    else:
                        safest_direction = 'LEFT'
                        safest_score = side_dangers['LEFT']
        
        # Calculate confidence (lower danger = higher confidence)
        max_danger = max(danger_scores.values()) if danger_scores.values() else 1
        confidence = 1.0 - (safest_score / max_danger) if max_danger > 0 else 1.0
        
        if safest_score == 0:
            status = 'REDIRECT_SAFE'
            recommendation = f"Obstacle in center - safe path available {safest_direction.lower()}"
        elif safest_score < center_danger:
            status = 'REDIRECT_CAUTION'
            recommendation = f"Multiple obstacles - proceed {safest_direction.lower()} with caution"
        else:
            status = 'DANGER_STOP'
            recommendation = "Obstacles in all directions - STOP and reassess"
            safest_direction = 'STOP'
            confidence = 0.0
        
        return {
            'status': status,
            'direction': safest_direction,
            'confidence': confidence,
            'obstacles_by_zone': zone_counts,
            'danger_scores': danger_scores,
            'recommendation': recommendation
        }
    
    def draw_navigation_overlay(self, frame, navigation_result, detections):
        """Draw navigation guidance on the frame"""
        
        # Draw zone boundaries
        self._draw_zones(frame)
        
        # Draw navigation arrow and status
        self._draw_navigation_arrow(frame, navigation_result)
        
        # Draw obstacle analysis
        self._draw_obstacle_analysis(frame, navigation_result, detections)
        
        return frame
    
    def _draw_zones(self, frame):
        """Draw navigation zone boundaries"""
        h, w = frame.shape[:2]
        
        # Draw grid lines
        cv2.line(frame, (w//3, 0), (w//3, h), (100, 100, 100), 1)
        cv2.line(frame, (2*w//3, 0), (2*w//3, h), (100, 100, 100), 1)
        cv2.line(frame, (0, h//3), (w, h//3), (100, 100, 100), 1)
        cv2.line(frame, (0, 2*h//3), (w, 2*h//3), (100, 100, 100), 1)
        
        # Label center zone
        cv2.putText(frame, "CENTER", (w//2-30, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_navigation_arrow(self, frame, nav_result):
        """Draw directional arrow based on navigation result"""
        h, w = frame.shape[:2]
        center = (w//2, h//2)
        
        direction = nav_result['direction']
        status = nav_result['status']
        
        # Color based on status
        if status == 'CLEAR_PATH':
            color = (0, 255, 0)  # Green
        elif status == 'REDIRECT_SAFE':
            color = (0, 255, 255)  # Yellow
        elif status == 'REDIRECT_CAUTION':
            color = (0, 165, 255)  # Orange
        else:  # DANGER_STOP
            color = (0, 0, 255)  # Red
        
        # Draw semi-transparent background circle for better visibility
        overlay = frame.copy()
        cv2.circle(overlay, center, 60, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw arrow based on direction
        arrow_length = 70
        if direction == 'FORWARD':
            end_point = (center[0], center[1] - arrow_length)
        elif direction == 'LEFT':
            end_point = (center[0] - arrow_length, center[1])
        elif direction == 'RIGHT':
            end_point = (center[0] + arrow_length, center[1])
        elif direction == 'UP':
            end_point = (center[0], center[1] - arrow_length)
        elif direction == 'DOWN':
            end_point = (center[0], center[1] + arrow_length)
        else:  # STOP
            # Draw stop sign with hexagon shape
            cv2.circle(frame, center, 40, color, -1)
            cv2.circle(frame, center, 40, (255, 255, 255), 3)
            cv2.putText(frame, "STOP", (center[0]-35, center[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
            return
        
        # Draw thick arrow with outline
        cv2.arrowedLine(frame, center, end_point, (255, 255, 255), 8, tipLength=0.4)
        cv2.arrowedLine(frame, center, end_point, color, 6, tipLength=0.4)
        
        # Draw direction text below arrow
        text_size = cv2.getTextSize(direction, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + 80
        cv2.putText(frame, direction, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
    
    def _draw_obstacle_analysis(self, frame, nav_result, detections):
        """Draw obstacle analysis information"""
        h, w = frame.shape[:2]
        
        status_text = nav_result['status'].replace('_', ' ')
        confidence = nav_result['confidence']
        recommendation = nav_result['recommendation']
        
        # Color based on status
        if nav_result['status'] == 'CLEAR_PATH':
            status_color = (0, 255, 0)  # Green
        elif nav_result['status'] == 'REDIRECT_SAFE':
            status_color = (0, 255, 255)  # Yellow
        elif nav_result['status'] == 'REDIRECT_CAUTION':
            status_color = (0, 165, 255)  # Orange
        else:
            status_color = (0, 0, 255)  # Red
        
        # Create semi-transparent panel for status info
        panel_height = 140
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-350, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Status text
        cv2.putText(frame, "STATUS:", (w-340, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, status_text, (w-340, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Confidence bar
        cv2.putText(frame, "CONFIDENCE:", (w-340, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        bar_width = int(300 * confidence)
        conf_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.3 else (0, 0, 255)
        cv2.rectangle(frame, (w-340, 100), (w-40, 120), (50, 50, 50), -1)
        cv2.rectangle(frame, (w-340, 100), (w-340+bar_width, 120), conf_color, -1)
        cv2.putText(frame, f"{confidence:.0%}", (w-120, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Recommendation at bottom with better visibility
        # Split long text into multiple lines
        max_chars = 60
        words = recommendation.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + word) <= max_chars:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
        
        # Draw recommendation with dark background
        line_height = 25
        total_height = len(lines) * line_height + 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, h - total_height - 5), (w - 5, h - 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, "RECOMMENDATION:", (10, h - total_height + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        for i, line in enumerate(lines):
            y_pos = h - total_height + 18 + (i + 1) * line_height
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def get_navigation_command(nav_result):
    """Convert navigation result to drone command format"""
    direction_map = {
        'FORWARD': {'x': 0, 'y': 1, 'z': 0},
        'LEFT': {'x': -1, 'y': 0, 'z': 0},
        'RIGHT': {'x': 1, 'y': 0, 'z': 0},
        'UP': {'x': 0, 'y': 0, 'z': 1},
        'DOWN': {'x': 0, 'y': 0, 'z': -1},
        'STOP': {'x': 0, 'y': 0, 'z': 0}
    }
    
    direction = nav_result['direction']
    confidence = nav_result['confidence']
    
    command = direction_map.get(direction, {'x': 0, 'y': 0, 'z': 0})
    
    # Scale movement by confidence
    speed_factor = confidence * 0.5  # Max speed 50% when fully confident
    
    return {
        'movement': {
            'x': command['x'] * speed_factor,
            'y': command['y'] * speed_factor,
            'z': command['z'] * speed_factor
        },
        'action': direction,
        'confidence': confidence,
        'should_stop': direction == 'STOP'
    }