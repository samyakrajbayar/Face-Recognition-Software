#!/usr/bin/env python3
"""
Advanced Face Recognition System
Complete implementation with all core classes and functionality
"""

import cv2
import face_recognition
import numpy as np
import sqlite3
import pickle
import os
import sys
import argparse
import time
import csv
import json
import yaml
import logging
from datetime import datetime, date
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import threading
import queue
from dataclasses import dataclass

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("Warning: GUI dependencies not available. GUI functionality will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_recognition.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Data class for face detection results."""
    name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (top, right, bottom, left)
    encoding: np.ndarray


class ConfigManager:
    """Configuration management for the application."""
    
    DEFAULT_CONFIG = {
        'recognition': {
            'tolerance': 0.6,
            'model': 'hog',  # 'hog' or 'cnn'
            'max_faces_per_frame': 10,
            'face_detection_scale': 0.25,
            'num_jitters': 1,
            'upsample_times': 1
        },
        'database': {
            'type': 'sqlite',
            'path': 'faces.db',
            'backup_enabled': True,
            'backup_interval': 86400,
            'max_backup_files': 5
        },
        'video': {
            'camera_index': 0,
            'frame_width': 640,
            'frame_height': 480,
            'fps': 30,
            'buffer_size': 1,
            'display_fps': True
        },
        'performance': {
            'process_every_n_frames': 1,
            'resize_factor': 0.25,
            'use_gpu': False,
            'max_threads': 4,
            'enable_optimization': True
        },
        'attendance': {
            'enabled': False,
            'log_file': 'attendance.csv',
            'prevent_duplicates': True,
            'duplicate_threshold': 300,  # seconds
            'custom_fields': [],
            'export_formats': ['csv', 'json']
        },
        'gui': {
            'theme': 'default',
            'window_size': '1000x700',
            'video_display_size': '640x480',
            'auto_save_settings': True,
            'show_confidence': True,
            'show_fps': True
        },
        'security': {
            'encrypt_database': False,
            'require_confidence': 70.0,
            'max_unknown_threshold': 10,
            'enable_face_tracking': True,
            'alert_on_unknown': False
        },
        'logging': {
            'level': 'INFO',
            'file': 'face_recognition.log',
            'max_file_size': 10485760,
            'backup_count': 5,
            'log_detections': True
        },
        'export': {
            'image_format': 'jpg',
            'video_codec': 'mp4v',
            'compression_quality': 85,
            'include_metadata': True
        }
    }
    
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    if user_config:
                        # Merge with defaults
                        config = self.DEFAULT_CONFIG.copy()
                        self._deep_update(config, user_config)
                        logger.info(f"Configuration loaded from {self.config_path}")
                        return config
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")
        
        # Create default config file
        self.save_config(self.DEFAULT_CONFIG)
        logger.info(f"Default configuration created at {self.config_path}")
        return self.DEFAULT_CONFIG.copy()
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Save configuration to file."""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def _deep_update(self, target: Dict, source: Dict) -> None:
        """Deep update dictionary."""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        target = self.config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value


class FaceDatabase:
    """Face encoding database management with SQLite backend."""
    
    def __init__(self, db_path: str = 'faces.db'):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database with proper schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create faces table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    encoding BLOB NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP,
                    recognition_count INTEGER DEFAULT 0
                )
            ''')
            
            # Create attendance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL NOT NULL,
                    source TEXT DEFAULT 'webcam',
                    metadata TEXT DEFAULT '{}'
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_updated ON faces(updated_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_name ON attendance(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_timestamp ON attendance(timestamp)')
            
            # Create triggers for updated_at
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS update_faces_timestamp 
                AFTER UPDATE ON faces
                BEGIN
                    UPDATE faces SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized successfully: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise RuntimeError(f"Failed to initialize database: {e}")
    
    def save_face(self, name: str, encoding: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """Save face encoding to database."""
        if encoding is None or len(encoding) == 0:
            logger.warning("Invalid encoding provided")
            return False
        
        if not name or not name.strip():
            logger.warning("Invalid name provided")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Serialize encoding and metadata
            encoding_blob = pickle.dumps(encoding.astype(np.float64))
            metadata_json = json.dumps(metadata or {})
            
            # Insert or update
            cursor.execute('''
                INSERT OR REPLACE INTO faces (name, encoding, metadata, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (name.strip(), encoding_blob, metadata_json))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Face encoding saved for: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving face encoding for {name}: {e}")
            return False
    
    def load_faces(self) -> Tuple[List[np.ndarray], List[str], List[Dict[str, Any]]]:
        """Load all face encodings, names, and metadata from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT name, encoding, metadata, recognition_count 
                FROM faces ORDER BY name
            ''')
            results = cursor.fetchall()
            
            conn.close()
            
            names = []
            encodings = []
            metadata_list = []
            
            for name, encoding_blob, metadata_json, recognition_count in results:
                try:
                    encoding = pickle.loads(encoding_blob)
                    metadata = json.loads(metadata_json or '{}')
                    metadata['recognition_count'] = recognition_count
                    
                    encodings.append(encoding)
                    names.append(name)
                    metadata_list.append(metadata)
                    
                except Exception as e:
                    logger.warning(f"Error loading encoding for {name}: {e}")
                    continue
            
            logger.info(f"Loaded {len(names)} face encodings from database")
            return encodings, names, metadata_list
            
        except Exception as e:
            logger.error(f"Error loading faces from database: {e}")
            return [], [], []
    
    def update_recognition_stats(self, name: str) -> bool:
        """Update recognition statistics for a face."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE faces 
                SET last_seen = CURRENT_TIMESTAMP,
                    recognition_count = recognition_count + 1
                WHERE name = ?
            ''', (name,))
            
            updated = cursor.rowcount > 0
            conn.commit()
            conn.close()
            
            return updated
            
        except Exception as e:
            logger.error(f"Error updating recognition stats for {name}: {e}")
            return False
    
    def delete_face(self, name: str) -> bool:
        """Delete face from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM faces WHERE name = ?', (name,))
            deleted = cursor.rowcount > 0
            
            conn.commit()
            conn.close()
            
            if deleted:
                logger.info(f"Face deleted: {name}")
            else:
                logger.warning(f"Face not found for deletion: {name}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting face {name}: {e}")
            return False
    
    def list_faces(self) -> List[Dict[str, Any]]:
        """List all faces in database with metadata."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, created_at, updated_at, last_seen, recognition_count, metadata
                FROM faces ORDER BY name
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            faces = []
            for row in results:
                face_id, name, created_at, updated_at, last_seen, recognition_count, metadata_json = row
                try:
                    metadata = json.loads(metadata_json or '{}')
                except:
                    metadata = {}
                
                faces.append({
                    'id': face_id,
                    'name': name,
                    'created_at': created_at,
                    'updated_at': updated_at,
                    'last_seen': last_seen,
                    'recognition_count': recognition_count,
                    'metadata': metadata
                })
            
            return faces
            
        except Exception as e:
            logger.error(f"Error listing faces: {e}")
            return []
    
    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """Create database backup."""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = f"{self.db_path}.backup_{timestamp}"
            
            # Use SQLite backup API for consistent backup
            source = sqlite3.connect(self.db_path)
            backup = sqlite3.connect(backup_path)
            source.backup(backup)
            
            source.close()
            backup.close()
            
            logger.info(f"Database backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return False


class AttendanceLogger:
    """Enhanced attendance logging with multiple export formats."""
    
    def __init__(self, log_file: str = 'attendance.csv', config: Optional[ConfigManager] = None):
        self.log_file = log_file
        self.config = config
        self.logged_today = {}  # name -> last_logged_timestamp
        self._init_log_file()
    
    def _init_log_file(self) -> None:
        """Initialize CSV log file with headers."""
        if not os.path.exists(self.log_file):
            try:
                headers = ['Date', 'Time', 'Name', 'Status', 'Confidence', 'Source']
                
                # Add custom fields if configured
                if self.config:
                    custom_fields = self.config.get('attendance.custom_fields', [])
                    headers.extend(custom_fields)
                
                with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                
                logger.info(f"Attendance log initialized: {self.log_file}")
                
            except Exception as e:
                logger.error(f"Error creating attendance log file: {e}")
    
    def log_attendance(self, name: str, status: str = 'Present', 
                      confidence: float = 0.0, source: str = 'webcam',
                      custom_data: Optional[Dict[str, Any]] = None) -> bool:
        """Log attendance entry with duplicate prevention."""
        now = datetime.now()
        today = now.date().isoformat()
        
        # Check for duplicates
        if self._is_duplicate(name, now):
            return False
        
        try:
            row_data = [
                today,
                now.strftime('%H:%M:%S'),
                name,
                status,
                f"{confidence:.1f}%",
                source
            ]
            
            # Add custom fields
            if self.config and custom_data:
                custom_fields = self.config.get('attendance.custom_fields', [])
                for field in custom_fields:
                    row_data.append(custom_data.get(field, ''))
            
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
            
            # Update duplicate prevention
            self.logged_today[name] = now.timestamp()
            
            logger.info(f"Attendance logged: {name} - {status} ({confidence:.1f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Error logging attendance for {name}: {e}")
            return False
    
    def _is_duplicate(self, name: str, current_time: datetime) -> bool:
        """Check if this is a duplicate entry."""
        if not self.config or not self.config.get('attendance.prevent_duplicates', True):
            return False
        
        if name not in self.logged_today:
            return False
        
        threshold = self.config.get('attendance.duplicate_threshold', 300)  # seconds
        last_logged = self.logged_today[name]
        time_diff = current_time.timestamp() - last_logged
        
        return time_diff < threshold
    
    def get_today_attendance(self) -> List[Dict[str, Any]]:
        """Get today's attendance records."""
        today = date.today().isoformat()
        attendance = []
        
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('Date') == today:
                            attendance.append(dict(row))
        except Exception as e:
            logger.error(f"Error reading attendance records: {e}")
        
        return attendance
    
    def export_attendance(self, start_date: Optional[str] = None, 
                         end_date: Optional[str] = None, 
                         format: str = 'csv') -> Optional[str]:
        """Export attendance data in specified format."""
        try:
            records = self._get_attendance_range(start_date, end_date)
            
            if not records:
                logger.warning("No attendance records found for export")
                return None
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format.lower() == 'json':
                export_file = f"attendance_export_{timestamp}.json"
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(records, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'csv':
                export_file = f"attendance_export_{timestamp}.csv"
                if records:
                    with open(export_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=records[0].keys())
                        writer.writeheader()
                        writer.writerows(records)
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
            
            logger.info(f"Attendance exported to: {export_file}")
            return export_file
            
        except Exception as e:
            logger.error(f"Error exporting attendance: {e}")
            return None
    
    def _get_attendance_range(self, start_date: Optional[str], 
                            end_date: Optional[str]) -> List[Dict[str, Any]]:
        """Get attendance records within date range."""
        records = []
        
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        record_date = row.get('Date', '')
                        
                        # Filter by date range if specified
                        if start_date and record_date < start_date:
                            continue
                        if end_date and record_date > end_date:
                            continue
                        
                        records.append(dict(row))
        
        except Exception as e:
            logger.error(f"Error reading attendance range: {e}")
        
        return records


class FaceRecognitionEngine:
    """Core face recognition engine with advanced features."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.tolerance = self.config.get('recognition.tolerance', 0.6)
        self.model = self.config.get('recognition.model', 'hog')
        
        # Initialize database and attendance
        db_path = self.config.get('database.path', 'faces.db')
        self.db = FaceDatabase(db_path)
        
        attendance_file = self.config.get('attendance.log_file', 'attendance.csv')
        self.attendance = AttendanceLogger(attendance_file, self.config)
        
        # Load known faces
        self.known_encodings, self.known_names, self.known_metadata = self.db.load_faces()
        
        # Performance tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Face tracking
        self.face_tracker = {} if self.config.get('security.enable_face_tracking', True) else None
        
        logger.info(f"Face recognition engine initialized with {len(self.known_names)} known faces")
    
    def detect_faces(self, image: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
        """Detect faces in image and return locations and encodings."""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Resize for faster processing if configured
            scale = self.config.get('recognition.face_detection_scale', 0.25)
            if scale != 1.0:
                h, w = rgb_image.shape[:2]
                small_image = cv2.resize(rgb_image, (int(w * scale), int(h * scale)))
            else:
                small_image = rgb_image
            
            # Detect faces
            upsample_times = self.config.get('recognition.upsample_times', 1)
            face_locations = face_recognition.face_locations(
                small_image, 
                model=self.model,
                number_of_times_to_upsample=upsample_times
            )
            
            # Scale back locations if image was resized
            if scale != 1.0:
                face_locations = [
                    (int(top/scale), int(right/scale), int(bottom/scale), int(left/scale))
                    for top, right, bottom, left in face_locations
                ]
            
            # Limit number of faces to process
            max_faces = self.config.get('recognition.max_faces_per_frame', 10)
            face_locations = face_locations[:max_faces]
            
            # Get face encodings
            num_jitters = self.config.get('recognition.num_jitters', 1)
            face_encodings = face_recognition.face_encodings(
                rgb_image, 
                face_locations,
                num_jitters=num_jitters
            )
            
            return face_locations, face_encodings
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return [], []
    
    def identify_face(self, face_encoding: np.ndarray) -> Tuple[str, float]:
        """Identify face from known faces database."""
        if len(self.known_encodings) == 0:
            return "Unknown", 0.0
        
        try:
            # Compare with known faces
            distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            best_match_index = np.argmin(distances)
            
            if distances[best_match_index] <= self.tolerance:
                name = self.known_names[best_match_index]
                confidence = (1 - distances[best_match_index]) * 100
                
                # Update recognition statistics
                self.db.update_recognition_stats(name)
                
                return name, confidence
            else:
                return "Unknown", 0.0
                
        except Exception as e:
            logger.error(f"Error in face identification: {e}")
            return "Unknown", 0.0
    
    def verify_face(self, face_encoding: np.ndarray, name: str) -> Tuple[bool, float]:
        """Verify if face encoding matches specific person."""
        try:
            if name not in self.known_names:
                return False, 0.0
            
            # Get the person's encoding
            person_index = self.known_names.index(name)
            person_encoding = self.known_encodings[person_index]
            
            # Compare encodings
            distance = face_recognition.face_distance([person_encoding], face_encoding)[0]
            confidence = (1 - distance) * 100
            
            is_match = distance <= self.tolerance
            
            if is_match:
                self.db.update_recognition_stats(name)
            
            return is_match, confidence
            
        except Exception as e:
            logger.error(f"Error in face verification: {e}")
            return False, 0.0
    
    def register_face(self, image: np.ndarray, name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Register new face in database."""
        try:
            if not name or not name.strip():
                logger.warning("Invalid name provided for face registration")
                return False
            
            face_locations, face_encodings = self.detect_faces(image)
            
            if len(face_encodings) == 0:
                logger.warning("No face detected in registration image")
                return False
            
            if len(face_encodings) > 1:
                logger.warning("Multiple faces detected. Please use image with single face.")
                return False
            
            # Save to database
            success = self.db.save_face(name.strip(), face_encodings[0], metadata)
            
            if success:
                # Reload known faces
                self.known_encodings, self.known_names, self.known_metadata = self.db.load_faces()
                logger.info(f"Successfully registered face: {name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error registering face for {name}: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, log_attendance: bool = False, 
                     source: str = 'webcam') -> Tuple[np.ndarray, List[DetectionResult]]:
        """Process single frame and return annotated frame with detection results."""
        try:
            # Update FPS counter
            self._update_fps()
            
            # Detect faces
            face_locations, face_encodings = self.detect_faces(frame)
            detection_results = []
            
            # Process each detected face
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Identify face
                name, confidence = self.identify_face(face_encoding)
                
                # Create detection result
                result = DetectionResult(
                    name=name,
                    confidence=confidence,
                    bbox=(top, right, bottom, left),
                    encoding=face_encoding
                )
                detection_results.append(result)
                
                # Log attendance if enabled and confidence meets threshold
                min_confidence = self.config.get('security.require_confidence', 70.0)
                if (log_attendance and name != "Unknown" and 
                    confidence >= min_confidence and 
                    self.config.get('attendance.enabled', False)):
                    
                    self.attendance.log_attendance(name, confidence=confidence, source=source)
                
                # Draw bounding box and label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                thickness = 2
                cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
                
                # Prepare label
                label = name
                if self.config.get('gui.show_confidence', True) and confidence > 0:
                    label += f" ({confidence:.1f}%)"
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
                cv2.rectangle(frame, (left, bottom - 35), (left + label_size[0], bottom), color, cv2.FILLED)
                
                # Draw label text
                cv2.putText(frame, label, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw FPS if enabled
            if self.config.get('gui.show_fps', True):
                fps_text = f"FPS: {self.current_fps:.1f}"
                cv2.putText(frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return frame, detection_results
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, []
    
    def _update_fps(self) -> None:
        """Update FPS calculation."""
        self.frame_count += 1
        if self.frame_count % 30 == 0:  # Update every 30 frames
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            if elapsed > 0:
                self.current_fps = 30 / elapsed
            self.fps_start_time = current_time
    
    def process_image_file(self, image_path: str, output_path: Optional[str] = None) -> bool:
        """Process image file and save results."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return False
            
            processed_image, results = self.process_frame(image)
            
            if output_path is None:
                # Generate output path
                path = Path(image_path)
                output_path = str(path.parent / f"{path.stem}_processed{path.suffix}")
            
            # Save processed image
            success = cv2.imwrite(output_path, processed_image)
            
            if success:
                logger.info(f"Processed image saved: {output_path}")
                logger.info(f"Detected {len(results)} faces")
                for result in results:
                    logger.info(f"  - {result.name} ({result.confidence:.1f}%)")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing image file: {e}")
            return False
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None, 
                          log_attendance: bool = False) -> bool:
        """Process video file and save results."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return False
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if output_path is None:
                path = Path(video_path)
                output_path = str(path.parent / f"{path.stem}_processed{path.suffix}")
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*self.config.get('export.video_codec', 'mp4v'))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            process_interval = self.config.get('performance.process_every_n_frames', 1)
            
            logger.info(f"Processing video: {total_frames} frames")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame based on interval
                if frame_count % process_interval == 0:
                    processed_frame, results = self.process_frame(
                        frame, 
                        log_attendance=log_attendance,
                        source=f'video:{video_path}'
                    )
                else:
                    processed_frame = frame
                
                # Write frame
                out.write(processed_frame)
                
                # Progress logging
                if frame_count % (fps * 10) == 0:  # Every 10 seconds
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            cap.release()
            out.release()
            
            logger.info(f"Processed video saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing video file: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get recognition system statistics."""
        try:
            faces = self.db.list_faces()
            
            stats = {
                'total_faces': len(faces),
                'total_recognitions': sum(face.get('recognition_count', 0) for face in faces),
                'most_recognized': None,
                'least_recognized': None,
                'recent_activity': [],
                'system_info': {
                    'tolerance': self.tolerance,
                    'model': self.model,
                    'current_fps': self.current_fps,
                    'total_frames_processed': self.frame_count
                }
            }
            
            if faces:
                # Sort by recognition count
                sorted_faces = sorted(faces, key=lambda x: x.get('recognition_count', 0), reverse=True)
                stats['most_recognized'] = {
                    'name': sorted_faces[0]['name'],
                    'count': sorted_faces[0].get('recognition_count', 0)
                }
                stats['least_recognized'] = {
                    'name': sorted_faces[-1]['name'],
                    'count': sorted_faces[-1].get('recognition_count', 0)
                }
                
                # Recent activity (faces seen in last 24 hours)
                from datetime import timedelta
                yesterday = datetime.now() - timedelta(days=1)
                
                for face in faces:
                    last_seen = face.get('last_seen')
                    if last_seen:
                        try:
                            last_seen_dt = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                            if last_seen_dt > yesterday:
                                stats['recent_activity'].append({
                                    'name': face['name'],
                                    'last_seen': last_seen,
                                    'recognition_count': face.get('recognition_count', 0)
                                })
                        except:
                            continue
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}


class FaceRecognitionGUI:
    """Advanced GUI interface for face recognition system."""
    
    def __init__(self):
        if not GUI_AVAILABLE:
            raise RuntimeError("GUI not available - required packages not installed")
        
        self.config = ConfigManager()
        self.engine = FaceRecognitionEngine(self.config)
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Advanced Face Recognition System v2.0")
        self.root.geometry(self.config.get('gui.window_size', '1000x700'))
        self.root.minsize(800, 600)
        
        # Variables
        self.is_webcam_running = False
        self.webcam_thread = None
        self.video_capture = None
        self.current_frame = None
        
        # GUI State
        self.face_list = None
        self.attendance_list = None
        
        self.setup_gui()
        self.setup_menu()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_menu(self):
        """Setup application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Process Image...", command=self.process_image_dialog)
        file_menu.add_command(label="Process Video...", command=self.process_video_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Export Attendance...", command=self.export_attendance_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Database menu
        db_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Database", menu=db_menu)
        db_menu.add_command(label="Register Face...", command=self.register_face_dialog)
        db_menu.add_command(label="View Database", command=self.view_database)
        db_menu.add_command(label="Delete Face...", command=self.delete_face_dialog)
        db_menu.add_separator()
        db_menu.add_command(label="Backup Database", command=self.backup_database)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Settings", command=self.settings_dialog)
        tools_menu.add_command(label="Statistics", command=self.statistics_dialog)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.about_dialog)
        
    def setup_gui(self):
        """Setup main GUI components."""
        # Create main container
        main_container = ttk.PanedWindow(self.root, orient='horizontal')
        main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left panel (controls)
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=1)
        
        # Right panel (video and info)
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=3)
        
        self.setup_control_panel(left_panel)
        self.setup_video_panel(right_panel)
        
    def setup_control_panel(self, parent):
        """Setup control panel."""
        # Webcam controls
        webcam_frame = ttk.LabelFrame(parent, text="Webcam Control", padding="10")
        webcam_frame.pack(fill='x', padx=5, pady=5)
        
        self.webcam_button = ttk.Button(webcam_frame, text="Start Webcam", 
                                       command=self.toggle_webcam)
        self.webcam_button.pack(fill='x', pady=2)
        
        # Attendance toggle
        self.attendance_var = tk.BooleanVar(value=self.config.get('attendance.enabled', False))
        attendance_cb = ttk.Checkbutton(webcam_frame, text="Log Attendance", 
                                      variable=self.attendance_var,
                                      command=self.toggle_attendance)
        attendance_cb.pack(anchor='w', pady=2)
        
        # Face management
        face_frame = ttk.LabelFrame(parent, text="Face Management", padding="10")
        face_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(face_frame, text="Register New Face", 
                  command=self.register_face_dialog).pack(fill='x', pady=2)
        ttk.Button(face_frame, text="View Database", 
                  command=self.view_database).pack(fill='x', pady=2)
        
        # Processing
        process_frame = ttk.LabelFrame(parent, text="File Processing", padding="10")
        process_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(process_frame, text="Process Image", 
                  command=self.process_image_dialog).pack(fill='x', pady=2)
        ttk.Button(process_frame, text="Process Video", 
                  command=self.process_video_dialog).pack(fill='x', pady=2)
        
        # Settings
        settings_frame = ttk.LabelFrame(parent, text="Quick Settings", padding="10")
        settings_frame.pack(fill='x', padx=5, pady=5)
        
        # Tolerance
        ttk.Label(settings_frame, text="Recognition Tolerance:").pack(anchor='w')
        self.tolerance_var = tk.DoubleVar(value=self.engine.tolerance)
        tolerance_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                                   variable=self.tolerance_var, orient='horizontal')
        tolerance_scale.pack(fill='x', pady=2)
        tolerance_scale.bind('<ButtonRelease-1>', self.update_tolerance)
        
        # Tolerance value display
        self.tolerance_label = ttk.Label(settings_frame, 
                                       text=f"Value: {self.tolerance_var.get():.2f}")
        self.tolerance_label.pack(anchor='w')
        
        # System info
        info_frame = ttk.LabelFrame(parent, text="System Info", padding="10")
        info_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.info_text = tk.Text(info_frame, height=6, wrap=tk.WORD, 
                                font=('Courier', 9))
        info_scrollbar = ttk.Scrollbar(info_frame, orient="vertical", 
                                     command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_text.pack(side='left', fill='both', expand=True)
        info_scrollbar.pack(side='right', fill='y')
        
        self.update_system_info()
        
    def setup_video_panel(self, parent):
        """Setup video display panel."""
        # Create notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Video tab
        video_frame = ttk.Frame(notebook)
        notebook.add(video_frame, text="Live Video")
        
        # Video canvas
        self.video_canvas = tk.Canvas(video_frame, bg='black')
        self.video_canvas.pack(fill='both', expand=True)
        
        # Detection info
        detection_frame = ttk.LabelFrame(video_frame, text="Recent Detections", padding="5")
        detection_frame.pack(fill='x', pady=(5, 0))
        
        self.detection_text = tk.Text(detection_frame, height=4, wrap=tk.WORD)
        det_scrollbar = ttk.Scrollbar(detection_frame, orient="vertical", 
                                    command=self.detection_text.yview)
        self.detection_text.configure(yscrollcommand=det_scrollbar.set)
        
        self.detection_text.pack(side='left', fill='both', expand=True)
        det_scrollbar.pack(side='right', fill='y')
        
        # Database tab
        db_frame = ttk.Frame(notebook)
        notebook.add(db_frame, text="Database")
        
        # Face list
        face_list_frame = ttk.LabelFrame(db_frame, text="Registered Faces", padding="5")
        face_list_frame.pack(fill='both', expand=True)
        
        # Treeview for faces
        columns = ('Name', 'Created', 'Last Seen', 'Count')
        self.face_tree = ttk.Treeview(face_list_frame, columns=columns, show='headings')
        
        for col in columns:
            self.face_tree.heading(col, text=col)
            self.face_tree.column(col, width=100)
        
        face_tree_scroll = ttk.Scrollbar(face_list_frame, orient="vertical", 
                                       command=self.face_tree.yview)
        self.face_tree.configure(yscrollcommand=face_tree_scroll.set)
        
        self.face_tree.pack(side='left', fill='both', expand=True)
        face_tree_scroll.pack(side='right', fill='y')
        
        # Attendance tab
        att_frame = ttk.Frame(notebook)
        notebook.add(att_frame, text="Attendance")
        
        attendance_list_frame = ttk.LabelFrame(att_frame, text="Today's Attendance", padding="5")
        attendance_list_frame.pack(fill='both', expand=True)
        
        # Treeview for attendance
        att_columns = ('Time', 'Name', 'Confidence', 'Status')
        self.attendance_tree = ttk.Treeview(attendance_list_frame, columns=att_columns, show='headings')
        
        for col in att_columns:
            self.attendance_tree.heading(col, text=col)
            self.attendance_tree.column(col, width=100)
        
        att_tree_scroll = ttk.Scrollbar(attendance_list_frame, orient="vertical", 
                                      command=self.attendance_tree.yview)
        self.attendance_tree.configure(yscrollcommand=att_tree_scroll.set)
        
        self.attendance_tree.pack(side='left', fill='both', expand=True)
        att_tree_scroll.pack(side='right', fill='y')
        
        # Refresh button
        ttk.Button(att_frame, text="Refresh Attendance", 
                  command=self.refresh_attendance).pack(pady=5)
        
        # Status bar
        self.status_bar = ttk.Label(parent, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(fill='x', side='bottom')
        
        # Initialize data
        self.refresh_database_view()
        self.refresh_attendance()
    
    def log_message(self, message: str, level: str = "INFO"):
        """Add message to detection log."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {level}: {message}"
        
        self.detection_text.insert(tk.END, formatted_message + "\n")
        self.detection_text.see(tk.END)
        
        # Limit text length
        lines = self.detection_text.get(1.0, tk.END).split('\n')
        if len(lines) > 100:
            self.detection_text.delete(1.0, f"{len(lines)-100}.0")
        
        self.root.update_idletasks()
    
    def update_status(self, message: str):
        """Update status bar."""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    def update_tolerance(self, event=None):
        """Update recognition tolerance."""
        new_tolerance = self.tolerance_var.get()
        self.engine.tolerance = new_tolerance
        self.tolerance_label.config(text=f"Value: {new_tolerance:.2f}")
        self.log_message(f"Tolerance updated to: {new_tolerance:.2f}")
    
    def toggle_attendance(self):
        """Toggle attendance logging."""
        enabled = self.attendance_var.get()
        self.config.config['attendance']['enabled'] = enabled
        self.config.save_config()
        status = "enabled" if enabled else "disabled"
        self.log_message(f"Attendance logging {status}")
    
    def toggle_webcam(self):
        """Toggle webcam recognition."""
        if not self.is_webcam_running:
            self.start_webcam()
        else:
            self.stop_webcam()
    
    def start_webcam(self):
        """Start webcam processing."""
        try:
            camera_index = self.config.get('video.camera_index', 0)
            self.video_capture = cv2.VideoCapture(camera_index)
            
            if not self.video_capture.isOpened():
                raise RuntimeError("Could not open camera")
            
            # Set video properties
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 
                                 self.config.get('video.frame_width', 640))
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 
                                 self.config.get('video.frame_height', 480))
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 
                                 self.config.get('video.buffer_size', 1))
            
            self.is_webcam_running = True
            self.webcam_thread = threading.Thread(target=self.webcam_loop, daemon=True)
            self.webcam_thread.start()
            
            self.webcam_button.config(text="Stop Webcam")
            self.update_status("Webcam started")
            self.log_message("Webcam started successfully")
            
        except Exception as e:
            error_msg = f"Error starting webcam: {e}"
            self.log_message(error_msg, "ERROR")
            self.update_status("Webcam failed to start")
            messagebox.showerror("Webcam Error", error_msg)
    
    def stop_webcam(self):
        """Stop webcam processing."""
        self.is_webcam_running = False
        
        if self.webcam_thread:
            self.webcam_thread.join(timeout=3)
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        # Clear canvas
        self.video_canvas.delete("all")
        self.video_canvas.configure(bg='black')
        
        self.webcam_button.config(text="Start Webcam")
        self.update_status("Webcam stopped")
        self.log_message("Webcam stopped")
    
    def webcam_loop(self):
        """Main webcam processing loop."""
        frame_count = 0
        last_detection_time = {}
        
        while self.is_webcam_running and self.video_capture:
            try:
                ret, frame = self.video_capture.read()
                if not ret:
                    break
                
                frame_count += 1
                self.current_frame = frame.copy()
                
                # Process every N frames for performance
                process_interval = self.config.get('performance.process_every_n_frames', 1)
                
                if frame_count % process_interval == 0:
                    processed_frame, results = self.engine.process_frame(
                        frame, 
                        log_attendance=self.attendance_var.get()
                    )
                    
                    # Log new detections
                    current_time = time.time()
                    for result in results:
                        if result.name != "Unknown":
                            # Avoid spam by limiting detection logging
                            if (result.name not in last_detection_time or 
                                current_time - last_detection_time[result.name] > 5):
                                
                                self.log_message(f"Detected: {result.name} ({result.confidence:.1f}%)")
                                last_detection_time[result.name] = current_time
                else:
                    processed_frame = frame
                
                # Display frame
                self.display_frame(processed_frame)
                
                # Update system info periodically
                if frame_count % 300 == 0:  # Every ~10 seconds at 30fps
                    self.root.after_idle(self.update_system_info)
                    if self.attendance_var.get():
                        self.root.after_idle(self.refresh_attendance)
                
            except Exception as e:
                logger.error(f"Error in webcam loop: {e}")
                break
        
        # Cleanup
        if self.is_webcam_running:
            self.root.after_idle(self.stop_webcam)
    
    def display_frame(self, frame):
        """Display frame in GUI canvas."""
        try:
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get canvas dimensions
            self.video_canvas.update()
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Calculate scaling while maintaining aspect ratio
                h, w = frame_rgb.shape[:2]
                scale = min(canvas_width/w, canvas_height/h)
                new_w, new_h = int(w*scale), int(h*scale)
                
                if new_w > 0 and new_h > 0:
                    frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
                    
                    # Convert to PIL format for tkinter
                    pil_image = Image.fromarray(frame_resized)
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # Update canvas
                    self.video_canvas.delete("all")
                    x = (canvas_width - new_w) // 2
                    y = (canvas_height - new_h) // 2
                    self.video_canvas.create_image(x, y, anchor=tk.NW, image=photo)
                    self.video_canvas.image = photo  # Keep reference
                
        except Exception as e:
            logger.error(f"Error displaying frame: {e}")
    
    def update_system_info(self):
        """Update system information display."""
        try:
            stats = self.engine.get_statistics()
            
            info_text = f"""Known Faces: {stats.get('total_faces', 0)}
Total Recognitions: {stats.get('total_recognitions', 0)}
Current FPS: {stats.get('system_info', {}).get('current_fps', 0.0):.1f}
Frames Processed: {stats.get('system_info', {}).get('total_frames_processed', 0)}
Model: {stats.get('system_info', {}).get('model', 'Unknown')}
Tolerance: {stats.get('system_info', {}).get('tolerance', 0.0):.2f}
"""
            
            if stats.get('most_recognized'):
                most = stats['most_recognized']
                info_text += f"Most Recognized: {most['name']} ({most['count']}x)\n"
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info_text)
            
        except Exception as e:
            logger.error(f"Error updating system info: {e}")
    
    def refresh_database_view(self):
        """Refresh the database view."""
        try:
            # Clear existing items
            for item in self.face_tree.get_children():
                self.face_tree.delete(item)
            
            # Load faces
            faces = self.engine.db.list_faces()
            
            for face in faces:
                created = face.get('created_at', 'Unknown')
                if created and created != 'Unknown':
                    try:
                        created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                        created = created_dt.strftime('%Y-%m-%d')
                    except:
                        pass
                
                last_seen = face.get('last_seen', 'Never')
                if last_seen and last_seen != 'Never':
                    try:
                        last_seen_dt = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                        last_seen = last_seen_dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        pass
                
                self.face_tree.insert('', 'end', values=(
                    face['name'],
                    created,
                    last_seen,
                    face.get('recognition_count', 0)
                ))
            
        except Exception as e:
            logger.error(f"Error refreshing database view: {e}")
    
    def refresh_attendance(self):
        """Refresh attendance view."""
        try:
            # Clear existing items
            for item in self.attendance_tree.get_children():
                self.attendance_tree.delete(item)
            
            # Load today's attendance
            attendance_records = self.engine.attendance.get_today_attendance()
            
            for record in attendance_records:
                self.attendance_tree.insert('', 'end', values=(
                    record.get('Time', ''),
                    record.get('Name', ''),
                    record.get('Confidence', ''),
                    record.get('Status', '')
                ))
            
        except Exception as e:
            logger.error(f"Error refreshing attendance: {e}")
    
    def register_face_dialog(self):
        """Open face registration dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Register New Face")
        dialog.geometry("450x350")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        
        # Center dialog
        dialog.geometry(f"+{self.root.winfo_rootx() + 50}+{self.root.winfo_rooty() + 50}")
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill='both', expand=True)
        
        # Name entry
        ttk.Label(main_frame, text="Person Name:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
        name_var = tk.StringVar()
        name_entry = ttk.Entry(main_frame, textvariable=name_var, width=40, font=('TkDefaultFont', 10))
        name_entry.pack(fill='x', pady=(5, 15))
        name_entry.focus()
        
        # Image selection
        ttk.Label(main_frame, text="Photo Selection:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
        
        # Option 1: From file
        file_frame = ttk.LabelFrame(main_frame, text="Option 1: From File", padding="10")
        file_frame.pack(fill='x', pady=(5, 10))
        
        image_path_var = tk.StringVar()
        path_frame = ttk.Frame(file_frame)
        path_frame.pack(fill='x')
        
        ttk.Entry(path_frame, textvariable=image_path_var, state='readonly').pack(side='left', fill='x', expand=True)
        ttk.Button(path_frame, text="Browse", 
                  command=lambda: self.browse_image(image_path_var)).pack(side='right', padx=(5, 0))
        
        # Option 2: From webcam
        webcam_frame = ttk.LabelFrame(main_frame, text="Option 2: From Webcam", padding="10")
        webcam_frame.pack(fill='x', pady=(0, 15))
        
        capture_button = ttk.Button(webcam_frame, text="Capture from Current Frame", 
                                  command=lambda: self.capture_current_frame(dialog, name_var),
                                  state='normal' if self.current_frame is not None else 'disabled')
        capture_button.pack(fill='x')
        
        if not self.is_webcam_running:
            ttk.Label(webcam_frame, text="Start webcam to enable capture", 
                     foreground='gray').pack(pady=(5, 0))
        
        # Metadata
        ttk.Label(main_frame, text="Additional Information (Optional):", 
                 font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
        metadata_frame = ttk.Frame(main_frame)
        metadata_frame.pack(fill='x', pady=(5, 15))
        
        ttk.Label(metadata_frame, text="Department:").grid(row=0, column=0, sticky='w', padx=(0, 10))
        dept_var = tk.StringVar()
        ttk.Entry(metadata_frame, textvariable=dept_var, width=20).grid(row=0, column=1, sticky='w')
        
        ttk.Label(metadata_frame, text="Role:").grid(row=0, column=2, sticky='w', padx=(20, 10))
        role_var = tk.StringVar()
        ttk.Entry(metadata_frame, textvariable=role_var, width=20).grid(row=0, column=3, sticky='w')
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(15, 0))
        
        ttk.Button(button_frame, text="Cancel", 
                  command=dialog.destroy).pack(side='right')
        ttk.Button(button_frame, text="Register Face", 
                  command=lambda: self.register_face_action(
                      dialog, name_var.get(), image_path_var.get(),
                      {'department': dept_var.get(), 'role': role_var.get()}
                  )).pack(side='right', padx=(0, 10))
        
        # Handle Enter key
        dialog.bind('<Return>', lambda e: self.register_face_action(
            dialog, name_var.get(), image_path_var.get(),
            {'department': dept_var.get(), 'role': role_var.get()}
        ))
    
    def browse_image(self, path_var):
        """Browse for image file."""
        filename = filedialog.askopenfilename(
            title="Select Photo",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            path_var.set(filename)
    
    def capture_current_frame(self, dialog, name_var):
        """Capture face from current webcam frame."""
        if self.current_frame is None:
            messagebox.showerror("Error", "No current frame available")
            return
        
        name = name_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name first")
            return
        
        # Save current frame temporarily
        temp_path = f"temp_capture_{int(time.time())}.jpg"
        try:
            cv2.imwrite(temp_path, self.current_frame)
            
            # Register the face
            if self.register_face_from_file(temp_path, name):
                dialog.destroy()
                self.log_message(f"Face registered from webcam capture: {name}")
                self.refresh_database_view()
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture frame: {e}")
    
    def register_face_action(self, dialog, name, image_path, metadata=None):
        """Register face action."""
        if not name or not name.strip():
            messagebox.showerror("Error", "Please enter a person name")
            return
        
        if not image_path:
            messagebox.showerror("Error", "Please select an image file")
            return
        
        if not os.path.exists(image_path):
            messagebox.showerror("Error", "Image file does not exist")
            return
        
        if self.register_face_from_file(image_path, name.strip(), metadata):
            dialog.destroy()
            self.log_message(f"Face registered successfully: {name}")
            self.refresh_database_view()
    
    def register_face_from_file(self, image_path, name, metadata=None):
        """Register face from image file."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                messagebox.showerror("Error", "Could not load image file")
                return False
            
            success = self.engine.register_face(image, name, metadata)
            
            if not success:
                messagebox.showerror("Error", 
                    "Failed to register face. Make sure the image contains exactly one clear face.")
                return False
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Registration failed: {e}")
            return False
    
    def delete_face_dialog(self):
        """Delete face dialog."""
        faces = self.engine.db.list_faces()
        
        if not faces:
            messagebox.showinfo("Information", "No faces registered in database")
            return
        
        # Create selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Delete Face")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        dialog.geometry(f"+{self.root.winfo_rootx() + 50}+{self.root.winfo_rooty() + 50}")
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill='both', expand=True)
        
        ttk.Label(main_frame, text="Select face to delete:", 
                 font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
        
        # Listbox with faces
        listbox_frame = ttk.Frame(main_frame)
        listbox_frame.pack(fill='both', expand=True, pady=(10, 15))
        
        listbox = tk.Listbox(listbox_frame)
        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        
        for face in faces:
            recognition_info = f" ({face.get('recognition_count', 0)} recognitions)"
            listbox.insert(tk.END, f"{face['name']}{recognition_info}")
        
        listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Warning
        ttk.Label(main_frame, text="⚠️ This action cannot be undone!", 
                 foreground='red').pack(pady=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x')
        
        def delete_selected():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a face to delete")
                return
            
            face_name = faces[selection[0]]['name']
            
            if messagebox.askyesno("Confirm Delete", 
                                  f"Are you sure you want to delete '{face_name}'?"):
                if self.engine.db.delete_face(face_name):
                    # Reload known faces in engine
                    self.engine.known_encodings, self.engine.known_names, self.engine.known_metadata = self.engine.db.load_faces()
                    
                    dialog.destroy()
                    self.log_message(f"Face deleted: {face_name}")
                    self.refresh_database_view()
                else:
                    messagebox.showerror("Error", f"Failed to delete face: {face_name}")
        
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side='right')
        ttk.Button(button_frame, text="Delete", command=delete_selected).pack(side='right', padx=(0, 10))
    
    def view_database(self):
        """Show database view dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Face Database")
        dialog.geometry("800x500")
        dialog.transient(self.root)
        
        dialog.geometry(f"+{self.root.winfo_rootx() + 25}+{self.root.winfo_rooty() + 25}")
        
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Database stats
        stats_frame = ttk.LabelFrame(main_frame, text="Database Statistics", padding="10")
        stats_frame.pack(fill='x', pady=(0, 10))
        
        stats = self.engine.get_statistics()
        stats_text = f"Total Faces: {stats.get('total_faces', 0)} | " \
                    f"Total Recognitions: {stats.get('total_recognitions', 0)}"
        
        if stats.get('most_recognized'):
            most = stats['most_recognized']
            stats_text += f" | Most Active: {most['name']} ({most['count']}x)"
        
        ttk.Label(stats_frame, text=stats_text).pack(anchor='w')
        
        # Face list
        list_frame = ttk.LabelFrame(main_frame, text="Registered Faces", padding="5")
        list_frame.pack(fill='both', expand=True)
        
        # Detailed treeview
        columns = ('Name', 'Department', 'Role', 'Created', 'Last Seen', 'Count')
        tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        
        # Configure columns
        tree.heading('Name', text='Name')
        tree.heading('Department', text='Department')
        tree.heading('Role', text='Role')
        tree.heading('Created', text='Created')
        tree.heading('Last Seen', text='Last Seen')
        tree.heading('Count', text='Recognitions')
        
        tree.column('Name', width=120)
        tree.column('Department', width=100)
        tree.column('Role', width=100)
        tree.column('Created', width=100)
        tree.column('Last Seen', width=140)
        tree.column('Count', width=80)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
        h_scrollbar = ttk.Scrollbar(list_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack widgets
        tree.pack(side='left', fill='both', expand=True)
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')
        
        # Load data
        faces = self.engine.db.list_faces()
        for face in faces:
            created = face.get('created_at', 'Unknown')
            if created and created != 'Unknown':
                try:
                    created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    created = created_dt.strftime('%Y-%m-%d')
                except:
                    pass
            
            last_seen = face.get('last_seen', 'Never')
            if last_seen and last_seen != 'Never':
                try:
                    last_seen_dt = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                    last_seen = last_seen_dt.strftime('%Y-%m-%d %H:%M')
                except:
                    pass
            
            metadata = face.get('metadata', {})
            department = metadata.get('department', '')
            role = metadata.get('role', '')
            
            tree.insert('', 'end', values=(
                face['name'],
                department,
                role,
                created,
                last_seen,
                face.get('recognition_count', 0)
            ))
        
        # Close button
        ttk.Button(main_frame, text="Close", command=dialog.destroy).pack(pady=(10, 0))
    
    def process_image_dialog(self):
        """Process image file dialog."""
        image_path = filedialog.askopenfilename(
            title="Select Image to Process",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if not image_path:
            return
        
        output_path = filedialog.asksaveasfilename(
            title="Save Processed Image As",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if not output_path:
            return
        
        try:
            self.update_status("Processing image...")
            success = self.engine.process_image_file(image_path, output_path)
            
            if success:
                self.log_message(f"Image processed: {output_path}")
                self.update_status("Image processing completed")
                
                if messagebox.askyesno("Success", 
                    f"Image processed successfully!\nSaved to: {output_path}\n\nOpen the processed image?"):
                    try:
                        if sys.platform.startswith('win'):
                            os.startfile(output_path)
                        elif sys.platform.startswith('darwin'):
                            os.system(f"open '{output_path}'")
                        else:
                            os.system(f"xdg-open '{output_path}'")
                    except:
                        pass
            else:
                messagebox.showerror("Error", "Failed to process image")
                self.update_status("Image processing failed")
                
        except Exception as e:
            error_msg = f"Error processing image: {e}"
            messagebox.showerror("Error", error_msg)
            self.log_message(error_msg, "ERROR")
            self.update_status("Ready")
    
    def process_video_dialog(self):
        """Process video file dialog."""
        video_path = filedialog.askopenfilename(
            title="Select Video to Process",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        
        if not video_path:
            return
        
        output_path = filedialog.asksaveasfilename(
            title="Save Processed Video As",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ]
        )
        
        if not output_path:
            return
        
        # Ask about attendance logging
        log_attendance = messagebox.askyesno(
            "Attendance Logging", 
            "Do you want to log attendance from this video?"
        )
        
        try:
            self.update_status("Processing video... This may take a while.")
            
            # Show progress dialog
            progress_dialog = self.show_progress_dialog("Processing Video", 
                                                      "Processing video file, please wait...")
            
            # Process in separate thread
            def process_video():
                try:
                    success = self.engine.process_video_file(video_path, output_path, log_attendance)
                    self.root.after_idle(lambda: self.video_processing_complete(
                        progress_dialog, success, output_path))
                except Exception as e:
                    self.root.after_idle(lambda: self.video_processing_error(progress_dialog, str(e)))
            
            threading.Thread(target=process_video, daemon=True).start()
            
        except Exception as e:
            error_msg = f"Error starting video processing: {e}"
            messagebox.showerror("Error", error_msg)
            self.log_message(error_msg, "ERROR")
            self.update_status("Ready")
    
    def show_progress_dialog(self, title, message):
        """Show progress dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("400x150")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        
        dialog.geometry(f"+{self.root.winfo_rootx() + 100}+{self.root.winfo_rooty() + 100}")
        
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        ttk.Label(main_frame, text=message).pack(pady=(0, 20))
        
        # Indeterminate progress bar
        progress = ttk.Progressbar(main_frame, mode='indeterminate')
        progress.pack(fill='x', pady=(0, 20))
        progress.start()
        
        ttk.Label(main_frame, text="This may take several minutes for large videos...", 
                 foreground='gray').pack()
        
        return dialog
    
    def video_processing_complete(self, progress_dialog, success, output_path):
        """Handle video processing completion."""
        progress_dialog.destroy()
        
        if success:
            self.log_message(f"Video processed: {output_path}")
            self.update_status("Video processing completed")
            self.refresh_attendance()
            
            if messagebox.askyesno("Success", 
                f"Video processed successfully!\nSaved to: {output_path}\n\nOpen the output folder?"):
                try:
                    output_dir = os.path.dirname(output_path)
                    if sys.platform.startswith('win'):
                        os.startfile(output_dir)
                    elif sys.platform.startswith('darwin'):
                        os.system(f"open '{output_dir}'")
                    else:
                        os.system(f"xdg-open '{output_dir}'")
                except:
                    pass
        else:
            messagebox.showerror("Error", "Failed to process video")
            self.update_status("Video processing failed")
    
    def video_processing_error(self, progress_dialog, error_message):
        """Handle video processing error."""
        progress_dialog.destroy()
        messagebox.showerror("Error", f"Video processing failed: {error_message}")
        self.log_message(f"Video processing error: {error_message}", "ERROR")
        self.update_status("Ready")
    
    def export_attendance_dialog(self):
        """Export attendance dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Attendance")
        dialog.geometry("400x250")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        
        dialog.geometry(f"+{self.root.winfo_rootx() + 100}+{self.root.winfo_rooty() + 100}")
        
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Date range
        ttk.Label(main_frame, text="Date Range:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
        
        date_frame = ttk.Frame(main_frame)
        date_frame.pack(fill='x', pady=(5, 15))
        
        ttk.Label(date_frame, text="From:").grid(row=0, column=0, sticky='w')
        start_date_var = tk.StringVar(value=date.today().strftime('%Y-%m-%d'))
        ttk.Entry(date_frame, textvariable=start_date_var, width=12).grid(row=0, column=1, padx=(5, 20))
        
        ttk.Label(date_frame, text="To:").grid(row=0, column=2, sticky='w')
        end_date_var = tk.StringVar(value=date.today().strftime('%Y-%m-%d'))
        ttk.Entry(date_frame, textvariable=end_date_var, width=12).grid(row=0, column=3, padx=(5, 0))
        
        # Format selection
        ttk.Label(main_frame, text="Export Format:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
        
        format_var = tk.StringVar(value="csv")
        format_frame = ttk.Frame(main_frame)
        format_frame.pack(fill='x', pady=(5, 20))
        
        ttk.Radiobutton(format_frame, text="CSV", variable=format_var, value="csv").pack(side='left')
        ttk.Radiobutton(format_frame, text="JSON", variable=format_var, value="json").pack(side='left', padx=(20, 0))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x')
        
        def export_attendance():
            try:
                start_date = start_date_var.get() if start_date_var.get() else None
                end_date = end_date_var.get() if end_date_var.get() else None
                format_type = format_var.get()
                
                export_file = self.engine.attendance.export_attendance(start_date, end_date, format_type)
                
                if export_file:
                    dialog.destroy()
                    self.log_message(f"Attendance exported: {export_file}")
                    
                    if messagebox.askyesno("Success", 
                        f"Attendance exported successfully!\nFile: {export_file}\n\nOpen the file?"):
                        try:
                            if sys.platform.startswith('win'):
                                os.startfile(export_file)
                            elif sys.platform.startswith('darwin'):
                                os.system(f"open '{export_file}'")
                            else:
                                os.system(f"xdg-open '{export_file}'")
                        except:
                            pass
                else:
                    messagebox.showwarning("Warning", "No attendance data found for the specified date range")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
        
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side='right')
        ttk.Button(button_frame, text="Export", command=export_attendance).pack(side='right', padx=(0, 10))
    
    def backup_database(self):
        """Backup database."""
        try:
            if self.engine.db.backup_database():
                self.log_message("Database backup created successfully")
                messagebox.showinfo("Success", "Database backup created successfully")
            else:
                messagebox.showerror("Error", "Failed to create database backup")
        except Exception as e:
            messagebox.showerror("Error", f"Backup failed: {e}")
    
    def settings_dialog(self):
        """Settings dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Settings")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        dialog.geometry(f"+{self.root.winfo_rootx() + 50}+{self.root.winfo_rooty() + 50}")
        
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Recognition settings
        recog_frame = ttk.Frame(notebook)
        notebook.add(recog_frame, text="Recognition")
        
        recog_content = ttk.Frame(recog_frame, padding="10")
        recog_content.pack(fill='both', expand=True)
        
        # Performance settings
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text="Performance")
        
        perf_content = ttk.Frame(perf_frame, padding="10")
        perf_content.pack(fill='both', expand=True)
        
        # Video settings
        video_frame = ttk.Frame(notebook)
        notebook.add(video_frame, text="Video")
        
        video_content = ttk.Frame(video_frame, padding="10")
        video_content.pack(fill='both', expand=True)
        
        # TODO: Add settings controls here
        ttk.Label(recog_content, text="Recognition settings will be added here").pack()
        ttk.Label(perf_content, text="Performance settings will be added here").pack()
        ttk.Label(video_content, text="Video settings will be added here").pack()
        
        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=(0, 10))
    
    def statistics_dialog(self):
        """Statistics dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("System Statistics")
        dialog.geometry("600x500")
        dialog.transient(self.root)
        
        dialog.geometry(f"+{self.root.winfo_rootx() + 25}+{self.root.winfo_rooty() + 25}")
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill='both', expand=True)
        
        # Get statistics
        stats = self.engine.get_statistics()
        
        # Create statistics display
        stats_text = tk.Text(main_frame, wrap=tk.WORD, font=('Courier', 10))
        stats_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=stats_text.yview)
        stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        # Format statistics
        content = f"""FACE RECOGNITION SYSTEM STATISTICS
=====================================

DATABASE STATISTICS:
• Total Registered Faces: {stats.get('total_faces', 0)}
• Total Recognition Events: {stats.get('total_recognitions', 0)}

SYSTEM PERFORMANCE:
• Current FPS: {stats.get('system_info', {}).get('current_fps', 0.0):.1f}
• Total Frames Processed: {stats.get('system_info', {}).get('total_frames_processed', 0):,}
• Recognition Model: {stats.get('system_info', {}).get('model', 'Unknown')}
• Recognition Tolerance: {stats.get('system_info', {}).get('tolerance', 0.0):.2f}

ACTIVITY SUMMARY:
"""
        
        if stats.get('most_recognized'):
            most = stats['most_recognized']
            content += f"• Most Recognized Person: {most['name']} ({most['count']} times)\n"
        
        if stats.get('least_recognized'):
            least = stats['least_recognized']
            content += f"• Least Recognized Person: {least['name']} ({least['count']} times)\n"
        
        content += f"\nRECENT ACTIVITY (Last 24 Hours):\n"
        recent = stats.get('recent_activity', [])
        if recent:
            for activity in recent[:10]:  # Show top 10
                content += f"• {activity['name']}: {activity['recognition_count']} recognitions\n"
        else:
            content += "• No recent activity recorded\n"
        
        # Today's attendance
        today_attendance = self.engine.attendance.get_today_attendance()
        content += f"\nTODAY'S ATTENDANCE: {len(today_attendance)} entries\n"
        
        for record in today_attendance[-10:]:  # Show last 10
            content += f"• {record.get('Time', '')}: {record.get('Name', '')} ({record.get('Confidence', '')})\n"
        
        stats_text.insert(1.0, content)
        stats_text.config(state='disabled')
        
        stats_text.pack(side='left', fill='both', expand=True)
        stats_scrollbar.pack(side='right', fill='y')
        
        # Refresh button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(button_frame, text="Refresh", command=lambda: self.refresh_statistics(stats_text)).pack(side='left')
        ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(side='right')
    
    def refresh_statistics(self, text_widget):
        """Refresh statistics display."""
        text_widget.config(state='normal')
        text_widget.delete(1.0, tk.END)
        
        # Get fresh statistics
        stats = self.engine.get_statistics()
        
        # Regenerate content (same as above)
        # ... (implementation similar to statistics_dialog)
        
        text_widget.config(state='disabled')
    
    def about_dialog(self):
        """About dialog."""
        messagebox.showinfo("About", 
            "Advanced Face Recognition System v2.0\n\n"
            "A comprehensive face recognition solution with:\n"
            "• Real-time face detection and recognition\n"
            "• Attendance logging and tracking\n"
            "• Database management\n"
            "• Batch processing capabilities\n\n"
            "Built with Python, OpenCV, and face_recognition library")
    
    def on_closing(self):
        """Handle application closing."""
        if self.is_webcam_running:
            self.stop_webcam()
        
        # Save configuration
        if self.config.get('gui.auto_save_settings', True):
            self.config.save_config()
        
        self.root.destroy()
    
    def run(self):
        """Run the GUI application."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        finally:
            self.on_closing()


class FaceRecognitionCLI:
    """Command-line interface for face recognition system."""
    
    def __init__(self):
        self.config = ConfigManager()
        self.engine = FaceRecognitionEngine(self.config)
    
    def run(self, args):
        """Run CLI based on arguments."""
        if args.command == 'register':
            return self.register_face(args.name, args.image, args.metadata)
        elif args.command == 'recognize':
            if args.webcam:
                return self.webcam_recognition(args.attendance)
            elif args.image:
                return self.recognize_image(args.image, args.output)
            elif args.video:
                return self.recognize_video(args.video, args.output, args.attendance)
        elif args.command == 'list':
            return self.list_faces()
        elif args.command == 'delete':
            return self.delete_face(args.name)
        elif args.command == 'stats':
            return self.show_statistics()
        elif args.command == 'export':
            return self.export_attendance(args.start_date, args.end_date, args.format)
        else:
            logger.error(f"Unknown command: {args.command}")
            return False
    
    def register_face(self, name: str, image_path: str, metadata: Optional[str] = None) -> bool:
        """Register face from CLI."""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return False
            
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return False
            
            # Parse metadata if provided
            meta_dict = {}
            if metadata:
                try:
                    meta_dict = json.loads(metadata)
                except json.JSONDecodeError:
                    # Try simple key=value format
                    for pair in metadata.split(','):
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            meta_dict[key.strip()] = value.strip()
            
            success = self.engine.register_face(image, name, meta_dict)
            
            if success:
                logger.info(f"Successfully registered face: {name}")
                return True
            else:
                logger.error("Face registration failed")
                return False
                
        except Exception as e:
            logger.error(f"Error registering face: {e}")
            return False
    
    def recognize_image(self, image_path: str, output_path: Optional[str] = None) -> bool:
        """Recognize faces in image."""
        try:
            success = self.engine.process_image_file(image_path, output_path)
            
            if success:
                logger.info("Image processing completed successfully")
                return True
            else:
                logger.error("Image processing failed")
                return False
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return False
    
    def recognize_video(self, video_path: str, output_path: Optional[str] = None, 
                       log_attendance: bool = False) -> bool:
        """Recognize faces in video."""
        try:
            success = self.engine.process_video_file(video_path, output_path, log_attendance)
            
            if success:
                logger.info("Video processing completed successfully")
                return True
            else:
                logger.error("Video processing failed")
                return False
                
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return False
    
    def webcam_recognition(self, log_attendance: bool = False) -> bool:
        """Run webcam recognition."""
        try:
            camera_index = self.config.get('video.camera_index', 0)
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                logger.error(f"Could not open camera {camera_index}")
                return False
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('video.frame_width', 640))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('video.frame_height', 480))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.get('video.buffer_size', 1))
            
            logger.info("Starting webcam recognition. Press 'q' to quit.")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                frame_count += 1
                
                # Process frame
                processed_frame, results = self.engine.process_frame(
                    frame, 
                    log_attendance=log_attendance
                )
                
                # Display results in console
                if results and frame_count % 30 == 0:  # Every 30 frames
                    for result in results:
                        if result.name != "Unknown":
                            logger.info(f"Detected: {result.name} ({result.confidence:.1f}%)")
                
                # Show frame
                cv2.imshow('Face Recognition', processed_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            logger.info("Webcam recognition stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error in webcam recognition: {e}")
            return False
    
    def list_faces(self) -> bool:
        """List all registered faces."""
        try:
            faces = self.engine.db.list_faces()
            
            if not faces:
                print("No faces registered in database")
                return True
            
            print(f"\nRegistered Faces ({len(faces)} total):")
            print("-" * 80)
            print(f"{'Name':<20} {'Created':<12} {'Last Seen':<20} {'Count':<8} {'Department':<15}")
            print("-" * 80)
            
            for face in faces:
                created = face.get('created_at', 'Unknown')
                if created and created != 'Unknown':
                    try:
                        created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                        created = created_dt.strftime('%Y-%m-%d')
                    except:
                        pass
                
                last_seen = face.get('last_seen', 'Never')
                if last_seen and last_seen != 'Never':
                    try:
                        last_seen_dt = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                        last_seen = last_seen_dt.strftime('%m-%d %H:%M')
                    except:
                        pass
                
                metadata = face.get('metadata', {})
                department = metadata.get('department', '')
                
                print(f"{face['name']:<20} {created:<12} {last_seen:<20} "
                      f"{face.get('recognition_count', 0):<8} {department:<15}")
            
            print("-" * 80)
            return True
            
        except Exception as e:
            logger.error(f"Error listing faces: {e}")
            return False
    
    def delete_face(self, name: str) -> bool:
        """Delete face from database."""
        try:
            # Confirm deletion
            response = input(f"Are you sure you want to delete '{name}'? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Deletion cancelled")
                return True
            
            success = self.engine.db.delete_face(name)
            
            if success:
                # Reload faces in engine
                self.engine.known_encodings, self.engine.known_names, self.engine.known_metadata = self.engine.db.load_faces()
                logger.info(f"Face deleted: {name}")
                return True
            else:
                logger.error(f"Face not found: {name}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting face: {e}")
            return False
    
    def show_statistics(self) -> bool:
        """Show system statistics."""
        try:
            stats = self.engine.get_statistics()
            
            print("\n" + "="*50)
            print("FACE RECOGNITION SYSTEM STATISTICS")
            print("="*50)
            
            print(f"\nDatabase Statistics:")
            print(f"  Total Faces: {stats.get('total_faces', 0)}")
            print(f"  Total Recognitions: {stats.get('total_recognitions', 0)}")
            
            print(f"\nSystem Performance:")
            sys_info = stats.get('system_info', {})
            print(f"  Current FPS: {sys_info.get('current_fps', 0.0):.1f}")
            print(f"  Frames Processed: {sys_info.get('total_frames_processed', 0):,}")
            print(f"  Recognition Model: {sys_info.get('model', 'Unknown')}")
            print(f"  Tolerance: {sys_info.get('tolerance', 0.0):.2f}")
            
            if stats.get('most_recognized'):
                most = stats['most_recognized']
                print(f"\nActivity Summary:")
                print(f"  Most Active: {most['name']} ({most['count']} recognitions)")
            
            recent = stats.get('recent_activity', [])
            if recent:
                print(f"\nRecent Activity (Last 24 Hours):")
                for activity in recent[:5]:
                    print(f"  {activity['name']}: {activity['recognition_count']} recognitions")
            
            # Today's attendance
            today_attendance = self.engine.attendance.get_today_attendance()
            print(f"\nToday's Attendance: {len(today_attendance)} entries")
            
            print("="*50)
            return True
            
        except Exception as e:
            logger.error(f"Error showing statistics: {e}")
            return False
    
    def export_attendance(self, start_date: Optional[str], end_date: Optional[str], 
                         format_type: str = 'csv') -> bool:
        """Export attendance data."""
        try:
            export_file = self.engine.attendance.export_attendance(start_date, end_date, format_type)
            
            if export_file:
                logger.info(f"Attendance exported to: {export_file}")
                return True
            else:
                logger.warning("No attendance data found for the specified date range")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting attendance: {e}")
            return False


def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Advanced Face Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start GUI
  python face_recognition_system.py gui

  # Register a face
  python face_recognition_system.py register --name "John Doe" --image person.jpg

  # Start webcam recognition with attendance logging
  python face_recognition_system.py recognize --webcam --attendance

  # Process an image
  python face_recognition_system.py recognize --image input.jpg --output output.jpg

  # List all registered faces
  python face_recognition_system.py list

  # Show system statistics
  python face_recognition_system.py stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Start GUI interface')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a new face')
    register_parser.add_argument('--name', required=True, help='Person name')
    register_parser.add_argument('--image', required=True, help='Path to image file')
    register_parser.add_argument('--metadata', help='Metadata in JSON format or key=value pairs')
    
    # Recognize command
    recognize_parser = subparsers.add_parser('recognize', help='Recognize faces')
    recognize_group = recognize_parser.add_mutually_exclusive_group(required=True)
    recognize_group.add_argument('--webcam', action='store_true', help='Use webcam')
    recognize_group.add_argument('--image', help='Process image file')
    recognize_group.add_argument('--video', help='Process video file')
    recognize_parser.add_argument('--output', help='Output file path')
    recognize_parser.add_argument('--attendance', action='store_true', help='Log attendance')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all registered faces')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a face')
    delete_parser.add_argument('--name', required=True, help='Name of person to delete')
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Show system statistics')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export attendance data')
    export_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    export_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    export_parser.add_argument('--format', choices=['csv', 'json'], default='csv', help='Export format')
    
    # Global options
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging level
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    
    try:
        if not args.command:
            # Default to GUI if available, otherwise show help
            if GUI_AVAILABLE:
                args.command = 'gui'
            else:
                parser.print_help()
                return 1
        
        if args.command == 'gui':
            if not GUI_AVAILABLE:
                print("Error: GUI not available. Install tkinter and PIL to use GUI mode.")
                return 1
            
            app = FaceRecognitionGUI()
            app.run()
            return 0
        
        else:
            # CLI commands
            cli = FaceRecognitionCLI()
            success = cli.run(args)
            return 0 if success else 1
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())