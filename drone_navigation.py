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
                # Factor in number and size of obstacles
                total_size = sum(obs['size'] for obs in obstacles)
                danger_scores[zone] = len(obstacles) * 0.7 + (total_size / 10000) * 0.3
        
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
        
        # Find safest direction
        side_dangers = {
            'LEFT': danger_scores['left'],
            'RIGHT': danger_scores['right'],
            'UP': danger_scores['top'],
            'DOWN': danger_scores['bottom']
        }
        
        safest_direction = min(side_dangers.keys(), key=lambda k: side_dangers[k])
        safest_score = side_dangers[safest_direction]
        
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
        
        # Draw arrow based on direction
        arrow_length = 50
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
            # Draw stop sign
            cv2.circle(frame, center, 30, color, 3)
            cv2.putText(frame, "STOP", (center[0]-20, center[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return
        
        # Draw arrow
        cv2.arrowedLine(frame, center, end_point, color, 4, tipLength=0.3)
        
        # Draw direction text
        cv2.putText(frame, direction, (center[0]-30, center[1]+70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    def _draw_obstacle_analysis(self, frame, nav_result, detections):
        """Draw obstacle analysis information"""
        h, w = frame.shape[:2]
        
        # Status text
        status_text = nav_result['status'].replace('_', ' ')
        cv2.putText(frame, f"STATUS: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Confidence
        confidence = nav_result['confidence']
        conf_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.3 else (0, 0, 255)
        cv2.putText(frame, f"CONFIDENCE: {confidence:.1%}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
        
        # Recommendation
        recommendation = nav_result['recommendation']
        cv2.putText(frame, recommendation[:50], (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if len(recommendation) > 50:
            cv2.putText(frame, recommendation[50:], (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


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