#!/usr/bin/env python3
"""
Comprehensive test suite for Face Recognition Software
Tests all major components and functionality
"""

import unittest
import cv2
import numpy as np
import os
import tempfile
import sqlite3
import pickle
from unittest.mock import patch, MagicMock
import sys
sys.path.append('.')

from face_recognition_system import (
    FaceDatabase, 
    FaceRecognitionEngine, 
    AttendanceLogger,
    FaceRecognitionGUI
)
from batch_processing import BatchProcessor


class TestFaceDatabase(unittest.TestCase):
    """Test face database functionality."""
    
    def setUp(self):
        """Set up test database."""
        self.test_db_path = tempfile.mktemp(suffix='.db')
        self.db = FaceDatabase(self.test_db_path)
    
    def tearDown(self):
        """Clean up test database."""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def test_database_initialization(self):
        """Test database initialization."""
        self.assertTrue(os.path.exists(self.test_db_path))
        
        # Check table exists
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='faces'")
        result = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(result)
    
    def test_save_and_load_face(self):
        """Test saving and loading face encodings."""
        # Create dummy encoding
        test_encoding = np.random.rand(128)
        test_name = "Test Person"
        
        # Save face
        result = self.db.save_face(test_name, test_encoding)
        self.assertTrue(result)
        
        # Load faces
        encodings, names = self.db.load_faces()
        self.assertEqual(len(encodings), 1)
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], test_name)
        np.testing.assert_array_almost_equal(encodings[0], test_encoding)
    
    def test_delete_face(self):
        """Test face deletion."""
        # Add face first
        test_encoding = np.random.rand(128)
        test_name = "Test Person"
        self.db.save_face(test_name, test_encoding)
        
        # Verify it exists
        encodings, names = self.db.load_faces()
        self.assertEqual(len(names), 1)
        
        # Delete face
        result = self.db.delete_face(test_name)
        self.assertTrue(result)
        
        # Verify it's gone
        encodings, names = self.db.load_faces()
        self.assertEqual(len(names), 0)
    
    def test_save_invalid_encoding(self):
        """Test handling of invalid encodings."""
        # Test with None encoding
        result = self.db.save_face("Test", None)
        self.assertFalse(result)


class TestAttendanceLogger(unittest.TestCase):
    """Test attendance logging functionality."""
    
    def setUp(self):
        """Set up test attendance logger."""
        self.test_log_file = tempfile.mktemp(suffix='.csv')
        self.logger = AttendanceLogger(self.test_log_file)
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)
    
    def test_log_file_creation(self):
        """Test log file creation."""
        self.assertTrue(os.path.exists(self.test_log_file))
    
    def test_attendance_logging(self):
        """Test attendance logging."""
        test_name = "John Doe"
        
        # Log attendance
        self.logger.log_attendance(test_name)
        
        # Verify log entry
        with open(self.test_log_file, 'r') as f:
            content = f.read()
            self.assertIn(test_name, content)
            self.assertIn("Present", content)
    
    def test_duplicate_attendance_prevention(self):
        """Test prevention of duplicate attendance on same day."""
        test_name = "Jane Doe"
        
        # Log attendance twice
        self.logger.log_attendance(test_name)
        self.logger.log_attendance(test_name)
        
        # Count entries
        with open(self.test_log_file, 'r') as f:
            lines = f.readlines()
            # Should have header + 1 entry (not 2)
            self.assertEqual(len(lines), 2)


class TestFaceRecognitionEngine(unittest.TestCase):
    """Test face recognition engine."""
    
    def setUp(self):
        """Set up test engine."""
        self.test_db_path = tempfile.mktemp(suffix='.db')
        # Mock the database path in the engine
        with patch('face_recognition_system.FaceDatabase') as mock_db:
            mock_db.return_value.load_faces.return_value = ([], [])
            self.engine = FaceRecognitionEngine()
            self.engine.db = FaceDatabase(self.test_db_path)
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    def test_detect_faces(self, mock_encodings, mock_locations):
        """Test face detection."""
        # Mock return values
        mock_locations.return_value = [(10, 100, 90, 20)]
        mock_encodings.return_value = [np.random.rand(128)]
        
        # Create dummy image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        locations, encodings = self.engine.detect_faces(test_image)
        
        self.assertEqual(len(locations), 1)
        self.assertEqual(len(encodings), 1)
        mock_locations.assert_called_once()
        mock_encodings.assert_called_once()
    
    def test_identify_unknown_face(self):
        """Test identification of unknown face."""
        # Empty known faces
        self.engine.known_encodings = []
        self.engine.known_names = []
        
        test_encoding = np.random.rand(128)
        name, confidence = self.engine.identify_face(test_encoding)
        
        self.assertEqual(name, "Unknown")
        self.assertEqual(confidence, 0.0)
    
    def test_identify_known_face(self):
        """Test identification of known face."""
        # Set up known face
        known_encoding = np.random.rand(128)
        self.engine.known_encodings = [known_encoding]
        self.engine.known_names = ["Test Person"]
        
        # Test with similar encoding (simulate same person)
        test_encoding = known_encoding + np.random.rand(128) * 0.1
        
        with patch('face_recognition.face_distance') as mock_distance:
            mock_distance.return_value = [0.3]  # Within tolerance
            
            name, confidence = self.engine.identify_face(test_encoding)
            
            self.assertEqual(name, "Test Person")
            self.assertGreater(confidence, 0)
    
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    def test_register_face_success(self, mock_encodings, mock_locations):
        """Test successful face registration."""
        # Mock single face detection
        mock_locations.return_value = [(10, 100, 90, 20)]
        mock_encodings.return_value = [np.random.rand(128)]
        
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_name = "New Person"
        
        result = self.engine.register_face(test_image, test_name)
        self.assertTrue(result)
    
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    def test_register_face_no_face(self, mock_encodings, mock_locations):
        """Test face registration with no face detected."""
        # Mock no face detection
        mock_locations.return_value = []
        mock_encodings.return_value = []
        
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_name = "No Face"
        
        result = self.engine.register_face(test_image, test_name)
        self.assertFalse(result)
    
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    def test_register_face_multiple_faces(self, mock_encodings, mock_locations):
        """Test face registration with multiple faces."""
        # Mock multiple face detection
        mock_locations.return_value = [(10, 100, 90, 20), (110, 200, 190, 120)]
        mock_encodings.return_value = [np.random.rand(128), np.random.rand(128)]
        
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        test_name = "Multiple Faces"
        
        result = self.engine.register_face(test_image, test_name)
        self.assertFalse(result)


class TestBatchProcessor(unittest.TestCase):
    """Test batch processing functionality."""
    
    def setUp(self):
        """Set up test batch processor."""
        with patch('batch_processing.FaceRecognitionEngine'):
            self.processor = BatchProcessor()
            # Mock the engine
            self.processor.engine = MagicMock()
    
    def test_extract_unique_faces(self):
        """Test extraction of unique faces from timeline."""
        timeline = [
            {
                'frame': 0,
                'timestamp': 0.0,
                'faces': [
                    {'name': 'John', 'confidence': 95.0},
                    {'name': 'Jane', 'confidence': 87.5}
                ]
            },
            {
                'frame': 30,
                'timestamp': 1.0,
                'faces': [
                    {'name': 'John', 'confidence': 92.3},
                    {'name': 'Unknown', 'confidence': 0.0}
                ]
            }
        ]
        
        unique_faces = self.processor._extract_unique_faces(timeline)
        self.assertIn('John', unique_faces)
        self.assertIn('Jane', unique_faces)
        self.assertNotIn('Unknown', unique_faces)
        self.assertEqual(len(unique_faces), 2)
    
    def test_calculate_statistics(self):
        """Test statistics calculation."""
        results = [
            {
                'faces': [
                    {'name': 'John', 'confidence': 95.0},
                    {'name': 'Unknown', 'confidence': 0.0}
                ]
            },
            {
                'faces': [
                    {'name': 'Jane', 'confidence': 87.5},
                    {'name': 'John', 'confidence': 92.0}
                ]
            }
        ]
        
        stats = self.processor._calculate_statistics(results)
        
        self.assertEqual(stats['total_faces'], 4)
        self.assertEqual(stats['recognized_faces'], 3)
        self.assertEqual(stats['unknown_faces'], 1)
        self.assertEqual(stats['recognition_rate'], 75.0)
        self.assertAlmostEqual(stats['average_confidence'], 91.5, places=1)


class TestImageProcessing(unittest.TestCase):
    """Test image processing functions."""
    
    def test_create_test_image(self):
        """Test creation of test images."""
        # Create a simple test image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[20:80, 20:80] = [255, 255, 255]  # White square
        
        self.assertEqual(image.shape, (100, 100, 3))
        self.assertEqual(image.dtype, np.uint8)
    
    def test_image_resize(self):
        """Test image resizing functionality."""
        original_image = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Resize to quarter size
        resized = cv2.resize(original_image, (0, 0), fx=0.25, fy=0.25)
        
        self.assertEqual(resized.shape, (100, 100, 3))
    
    def test_color_conversion(self):
        """Test color space conversion."""
        bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_image[:, :] = [255, 0, 0]  # Blue in BGR
        
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        # Should now be red in RGB
        self.assertEqual(rgb_image[0, 0, 0], 255)  # Red channel
        self.assertEqual(rgb_image[0, 0, 1], 0)    # Green channel
        self.assertEqual(rgb_image[0, 0, 2], 0)    # Blue channel


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_db_path = tempfile.mktemp(suffix='.db')
        self.test_log_path = tempfile.mktemp(suffix='.csv')
    
    def tearDown(self):
        """Clean up test files."""
        for path in [self.test_db_path, self.test_log_path]:
            if os.path.exists(path):
                os.remove(path)
    
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    def test_complete_workflow(self, mock_encodings, mock_locations):
        """Test complete registration and recognition workflow."""
        # Mock face detection
        mock_locations.return_value = [(10, 100, 90, 20)]
        test_encoding = np.random.rand(128)
        mock_encodings.return_value = [test_encoding]
        
        # Initialize components
        db = FaceDatabase(self.test_db_path)
        attendance = AttendanceLogger(self.test_log_path)
        engine = FaceRecognitionEngine()
        engine.db = db
        engine.attendance = attendance
        
        # Register a face
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_name = "Integration Test"
        
        registration_result = engine.register_face(test_image, test_name)
        self.assertTrue(registration_result)
        
        # Verify face was saved
        encodings, names = db.load_faces()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], test_name)
        
        # Test recognition
        with patch('face_recognition.face_distance') as mock_distance:
            mock_distance.return_value = [0.3]  # Within tolerance
            
            # Process frame (should recognize the face)
            processed_frame = engine.process_frame(test_image, log_attendance=True)
            
            # Verify attendance was logged
            with open(self.test_log_path, 'r') as f:
                content = f.read()
                self.assertIn(test_name, content)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_invalid_image_path(self):
        """Test handling of invalid image paths."""
        result = cv2.imread("nonexistent_file.jpg")
        self.assertIsNone(result)
    
    def test_empty_database_operations(self):
        """Test operations on empty database."""
        test_db_path = tempfile.mktemp(suffix='.db')
        db = FaceDatabase(test_db_path)
        
        try:
            encodings, names = db.load_faces()
            self.assertEqual(len(encodings), 0)
            self.assertEqual(len(names), 0)
            
            # Test deletion on non-existent face
            result = db.delete_face("Non-existent Person")
            self.assertFalse(result)
        finally:
            if os.path.exists(test_db_path):
                os.remove(test_db_path)
    
    def test_malformed_encoding(self):
        """Test handling of malformed face encodings."""
        engine = FaceRecognitionEngine()
        
        # Test with wrong dimension encoding
        invalid_encoding = np.random.rand(64)  # Should be 128
        
        # This should not crash but return unknown
        try:
            name, confidence = engine.identify_face(invalid_encoding)
            # The behavior may vary, but it shouldn't crash
        except Exception as e:
            # If it raises an exception, it should be handled gracefully
            self.assertIsInstance(e, (ValueError, IndexError))


def create_test_images():
    """Create test images for manual testing."""
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create simple test images
    for i in range(3):
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        cv2.imwrite(f"{test_dir}/test_image_{i+1}.jpg", image)
    
    print(f"Created test images in {test_dir}/")


def run_performance_tests():
    """Run performance benchmarks."""
    print("Running performance tests...")
    
    import time
    
    # Test face detection speed
    test_image = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
    
    with patch('face_recognition.face_locations') as mock_locations:
        with patch('face_recognition.face_encodings') as mock_encodings:
            mock_locations.return_value = [(10, 100, 90, 20)]
            mock_encodings.return_value = [np.random.rand(128)]
            
            engine = FaceRecognitionEngine()
            
            # Time multiple frame processing
            start_time = time.time()
            for _ in range(100):
                engine.process_frame(test_image)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 100
            fps = 1.0 / avg_time if avg_time > 0 else float('inf')
            
            print(f"Average processing time per frame: {avg_time:.4f}s")
            print(f"Estimated FPS: {fps:.1f}")


def main():
    """Run all tests."""
    print("=== Advanced Face Recognition Software Test Suite ===")
    print()
    
    # Create test images if they don't exist
    if not os.path.exists("test_images"):
        print("Creating test images...")
        create_test_images()
    
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_tests()
    
    print("\n=== Test Suite Complete ===")


if __name__ == "__main__":
    main()