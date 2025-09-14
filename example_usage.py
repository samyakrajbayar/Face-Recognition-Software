#!/usr/bin/env python3
"""
Example usage scripts for the Advanced Face Recognition Software
Demonstrates various use cases and integrations
"""

import cv2
import numpy as np
import time
import json
from datetime import datetime
import sys
import os

# Add the main module to path
sys.path.append('.')
from face_recognition_system import FaceRecognitionEngine, FaceDatabase, AttendanceLogger

# Example 1: Basic Face Registration and Recognition
def example_basic_usage():
    """Basic usage example: register faces and recognize them."""
    print("=== Example 1: Basic Usage ===")
    
    # Initialize the engine
    engine = FaceRecognitionEngine(tolerance=0.6)
    
    # Create a synthetic face image (in real use, load actual photos)
    def create_mock_face(person_id):
        """Create a mock face image for testing."""
        image = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        # Add some structure to make it more face-like
        cv2.rectangle(image, (50, 50), (150, 150), (100, 100, 100), -1)
        cv2.circle(image, (80, 80), 10, (0, 0, 0), -1)  # Eye
        cv2.circle(image, (120, 80), 10, (0, 0, 0), -1)  # Eye
        cv2.rectangle(image, (90, 110), (110, 120), (0, 0, 0), -1)  # Nose
        cv2.rectangle(image, (75, 130), (125, 140), (0, 0, 0), -1)  # Mouth
        return image
    
    # Register some faces (in real use, use actual photos)
    people = ["Alice", "Bob", "Charlie"]
    for person in people:
        mock_image = create_mock_face(person)
        print(f"Registering {person}...")
        # Note: This will likely fail with mock images, but shows the API
        success = engine.register_face(mock_image, person)
        if success:
            print(f"✓ {person} registered successfully")
        else:
            print(f"✗ Failed to register {person}")
    
    print(f"Total registered faces: {len(engine.known_names)}")
    print("Registered names:", engine.known_names)


# Example 2: Attendance System
def example_attendance_system():
    """Example of using the system for attendance tracking."""
    print("\n=== Example 2: Attendance System ===")
    
    class AttendanceSystem:
        def __init__(self):
            self.engine = FaceRecognitionEngine(tolerance=0.5)
            self.attendance_log = AttendanceLogger("daily_attendance.csv")
            self.processed_today = set()
        
        def process_attendance(self, image):
            """Process attendance from an image."""
            locations, encodings = self.engine.detect_faces(image)
            
            for encoding in encodings:
                name, confidence = self.engine.identify_face(encoding)
                
                if name != "Unknown" and confidence > 80:
                    if name not in self.processed_today:
                        self.attendance_log.log_attendance(name, "Present")
                        self.processed_today.add(name)
                        print(f"✓ Marked {name} as present (confidence: {confidence:.1f}%)")
                    else:
                        print(f"- {name} already marked present today")
                elif name != "Unknown":
                    print(f"? Low confidence for {name} ({confidence:.1f}%)")
                else:
                    print("? Unknown person detected")
        
        def get_daily_report(self):
            """Get daily attendance report."""
            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "present_count": len(self.processed_today),
                "present_people": list(self.processed_today)
            }
    
    # Initialize attendance system
    attendance_sys = AttendanceSystem()
    
    # Simulate processing some images
    print("Processing attendance images...")
    for i in range(3):
        mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        attendance_sys.process_attendance(mock_image)
    
    # Get daily report
    report = attendance_sys.get_daily_report()
    print("Daily Report:", json.dumps(report, indent=2))


# Example 3: Security Access Control
def example_security_system():
    """Example of using the system for security access control."""
    print("\n=== Example 3: Security Access Control ===")
    
    class SecurityAccessControl:
        def __init__(self):
            self.engine = FaceRecognitionEngine(tolerance=0.4)  # Stricter for security
            self.authorized_personnel = {
                "Admin": {"access_level": 5, "areas": ["all"]},
                "Manager": {"access_level": 3, "areas": ["office", "meeting_room"]},
                "Employee": {"access_level": 1, "areas": ["office"]}
            }
            self.access_log = []
        
        def check_access(self, image, requested_area="office"):
            """Check if person in image has access to requested area."""
            locations, encodings = self.engine.detect_faces(image)
            
            if len(encodings) != 1:
                return self._log_access("Multiple/No faces detected", False, requested_area)
            
            name, confidence = self.engine.identify_face(encodings[0])
            
            if name == "Unknown" or confidence < 90:
                return self._log_access(f"Unknown/Low confidence ({confidence:.1f}%)", 
                                      False, requested_area)
            
            # Check authorization
            if name in self.authorized_personnel:
                person_info = self.authorized_personnel[name]
                has_access = (requested_area in person_info["areas"] or 
                            "all" in person_info["areas"])
                
                return self._log_access(name, has_access, requested_area, 
                                      person_info["access_level"])
            else:
                return self._log_access(name, False, requested_area, reason="Not authorized")
        
        def _log_access(self, person, granted, area, access_level=0, reason=""):
            """Log access attempt."""
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "person": person,
                "area": area,
                "access_granted": granted,
                "access_level": access_level,
                "reason": reason
            }
            self.access_log.append(log_entry)
            
            status = "GRANTED" if granted else "DENIED"
            print(f"{status}: {person} -> {area}")
            if reason:
                print(f"  Reason: {reason}")
            
            return granted
        
        def get_access_report(self):
            """Get access log report."""
            granted = sum(1 for log in self.access_log if log["access_granted"])
            denied = len(self.access_log) - granted
            
            return {
                "total_attempts": len(self.access_log),
                "granted": granted,
                "denied": denied,
                "recent_activity": self.access_log[-10:] if self.access_log else []
            }
    
    # Initialize security system
    security_sys = SecurityAccessControl()
    
    # Simulate access attempts
    print("Simulating access control scenarios...")
    areas = ["office", "meeting_room", "server_room"]
    
    for area in areas:
        print(f"\nChecking access to {area}:")
        mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        security_sys.check_access(mock_image, area)
    
    # Get security report
    report = security_sys.get_access_report()
    print("\nSecurity Report:", json.dumps(report, indent=2))


# Example 4: Real-time Webcam Processing
def example_realtime_webcam():
    """Example of real-time webcam processing with custom overlays."""
    print("\n=== Example 4: Real-time Webcam Processing ===")
    
    class RealtimeProcessor:
        def __init__(self):
            self.engine = FaceRecognitionEngine()
            self.frame_count = 0
            self.fps_counter = 0
            self.start_time = time.time()
            self.recognition_history = {}
        
        def process_webcam_feed(self, duration_seconds=10):
            """Process webcam feed for specified duration."""
            print(f"Starting webcam processing for {duration_seconds} seconds...")
            print("Note: This is a simulation - no actual webcam access")
            
            # Simulate webcam processing
            end_time = time.time() + duration_seconds
            
            while time.time() < end_time:
                # Simulate frame processing
                self.frame_count += 1
                
                # Mock frame
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # Process frame (normally would be actual processing)
                processed_frame = self.process_frame_with_stats(frame)
                
                # Simulate display
                time.sleep(0.033)  # ~30 FPS
                
                # Update FPS
                current_time = time.time()
                if current_time - self.start_time >= 1.0:
                    self.fps_counter = self.frame_count / (current_time - self.start_time)
                    print(f"Processing at {self.fps_counter:.1f} FPS")
                    self.frame_count = 0
                    self.start_time = current_time
            
            print("Webcam processing completed")
        
        def process_frame_with_stats(self, frame):
            """Process frame and add statistics overlay."""
            # Add frame counter
            cv2.putText(frame, f"Frame: {self.frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add FPS counter
            cv2.putText(frame, f"FPS: {self.fps_counter:.1f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return frame
    
    # Initialize real-time processor
    processor = RealtimeProcessor()
    
    # Process webcam feed (simulated)
    processor.process_webcam_feed(duration_seconds=5)


# Example 5: Batch Photo Organization
def example_photo_organization():
    """Example of organizing photos by recognized faces."""
    print("\n=== Example 5: Photo Organization ===")
    
    class PhotoOrganizer:
        def __init__(self):
            self.engine = FaceRecognitionEngine()
            self.photo_database = {}
        
        def organize_photos(self, photo_directory="photos/"):
            """Organize photos by detected faces."""
            print(f"Organizing photos from {photo_directory}")
            
            # Simulate photo directory
            mock_photos = [f"photo_{i:03d}.jpg" for i in range(1, 21)]
            
            for photo_name in mock_photos:
                print(f"Processing {photo_name}...")
                
                # Mock image processing
                mock_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
                
                # Simulate face detection and recognition
                detected_faces = self.analyze_photo(mock_image, photo_name)
                
                # Organize by faces
                for face_info in detected_faces:
                    person_name = face_info['name']
                    if person_name not in self.photo_database:
                        self.photo_database[person_name] = []
                    
                    self.photo_database[person_name].append({
                        'photo': photo_name,
                        'confidence': face_info['confidence'],
                        'location': face_info['location']
                    })
        
        def analyze_photo(self, image, photo_name):
            """Analyze photo for faces."""
            # Mock face detection results
            mock_faces = [
                {
                    'name': np.random.choice(['Alice', 'Bob', 'Charlie', 'Unknown']),
                    'confidence': np.random.uniform(70, 95),
                    'location': (50, 50, 150, 150)
                }
                for _ in range(np.random.randint(0, 3))
            ]
            
            return mock_faces
        
        def generate_organization_report(self):
            """Generate photo organization report."""
            report = {
                'total_people': len([k for k in self.photo_database.keys() if k != 'Unknown']),
                'total_photos_with_faces': sum(len(photos) for photos in self.photo_database.values()),
                'people_summary': {}
            }
            
            for person, photos in self.photo_database.items():
                if person != 'Unknown':
                    avg_confidence = sum(p['confidence'] for p in photos) / len(photos)
                    report['people_summary'][person] = {
                        'photo_count': len(photos),
                        'average_confidence': avg_confidence
                    }
            
            return report
    
    # Initialize photo organizer
    organizer = PhotoOrganizer()
    
    # Organize photos
    organizer.organize_photos()
    
    # Generate report
    report = organizer.generate_organization_report()
    print("Organization Report:", json.dumps(report, indent=2))


# Example 6: Custom Integration with External Systems
def example_external_integration():
    """Example of integrating with external systems (webhooks, APIs)."""
    print("\n=== Example 6: External System Integration ===")
    
    class ExternalIntegration:
        def __init__(self):
            self.engine = FaceRecognitionEngine()
            self.webhook_url = "https://api.example.com/face-detection"
            self.api_key = "your-api-key-here"
        
        def process_with_notifications(self, image):
            """Process image and send notifications to external systems."""
            locations, encodings = self.engine.detect_faces(image)
            
            results = []
            for i, encoding in enumerate(encodings):
                name, confidence = self.engine.identify_face(encoding)
                
                result = {
                    'face_id': i + 1,
                    'name': name,
                    'confidence': confidence,
                    'location': locations[i] if i < len(locations) else None,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'recognized' if name != 'Unknown' else 'unknown'
                }
                results.append(result)
                
                # Send notification for recognized faces
                if name != 'Unknown' and confidence > 80:
                    self.send_webhook_notification(result)
            
            return results
        
        def send_webhook_notification(self, face_data):
            """Send webhook notification (simulated)."""
            print(f"📧 Webhook notification sent for {face_data['name']}")
            print(f"   Confidence: {face_data['confidence']:.1f}%")
            print(f"   Timestamp: {face_data['timestamp']}")
            
            # In real implementation, you would use requests library:
            # import requests
            # payload = {
            #     'event': 'face_recognized',
            #     'data': face_data,
            #     'api_key': self.api_key
            # }
            # response = requests.post(self.webhook_url, json=payload)
        
        def integrate_with_database(self, face_data):
            """Integrate with external database (simulated)."""
            print(f"💾 Storing to external database:")
            print(f"   Person: {face_data['name']}")
            print(f"   Event: Face Recognition")
            print(f"   Confidence: {face_data['confidence']:.1f}%")
            
            # In real implementation, you would connect to your database:
            # import pymongo  # for MongoDB
            # import psycopg2  # for PostgreSQL
            # etc.
    
    # Initialize external integration
    integration = ExternalIntegration()
    
    # Process image with external notifications
    mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    results = integration.process_with_notifications(mock_image)
    
    print("External Integration Results:")
    for result in results:
        print(f"  - {result['name']}: {result['confidence']:.1f}% confidence")


# Example 7: Performance Monitoring and Optimization
def example_performance_monitoring():
    """Example of monitoring and optimizing performance."""
    print("\n=== Example 7: Performance Monitoring ===")
    
    class PerformanceMonitor:
        def __init__(self):
            self.engine = FaceRecognitionEngine()
            self.metrics = {
                'processing_times': [],
                'face_counts': [],
                'memory_usage': [],
                'confidence_scores': []
            }
        
        def benchmark_processing(self, num_images=50):
            """Benchmark face recognition processing."""
            print(f"Running benchmark with {num_images} images...")
            
            total_start_time = time.time()
            
            for i in range(num_images):
                # Create test image
                image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # Time the processing
                start_time = time.time()
                locations, encodings = self.engine.detect_faces(image)
                processing_time = time.time() - start_time
                
                # Record metrics
                self.metrics['processing_times'].append(processing_time)
                self.metrics['face_counts'].append(len(encodings))
                
                # Simulate recognition for detected faces
                for encoding in encodings:
                    name, confidence = self.engine.identify_face(encoding)
                    self.metrics['confidence_scores'].append(confidence)
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{num_images} images...")
            
            total_time = time.time() - total_start_time
            self.generate_performance_report(total_time)
        
        def generate_performance_report(self, total_time):
            """Generate detailed performance report."""
            processing_times = self.metrics['processing_times']
            
            report = {
                'benchmark_summary': {
                    'total_images': len(processing_times),
                    'total_time': f"{total_time:.2f}s",
                    'images_per_second': f"{len(processing_times) / total_time:.2f}",
                    'average_processing_time': f"{np.mean(processing_times):.4f}s"
                },
                'processing_stats': {
                    'min_processing_time': f"{np.min(processing_times):.4f}s",
                    'max_processing_time': f"{np.max(processing_times):.4f}s",
                    'median_processing_time': f"{np.median(processing_times):.4f}s",
                    'std_dev': f"{np.std(processing_times):.4f}s"
                },
                'face_detection': {
                    'total_faces_detected': sum(self.metrics['face_counts']),
                    'average_faces_per_image': f"{np.mean(self.metrics['face_counts']):.2f}",
                    'max_faces_in_single_image': max(self.metrics['face_counts']) if self.metrics['face_counts'] else 0
                }
            }
            
            if self.metrics['confidence_scores']:
                report['recognition_stats'] = {
                    'average_confidence': f"{np.mean(self.metrics['confidence_scores']):.1f}%",
                    'min_confidence': f"{np.min(self.metrics['confidence_scores']):.1f}%",
                    'max_confidence': f"{np.max(self.metrics['confidence_scores']):.1f}%"
                }
            
            print("\nPerformance Report:")
            print(json.dumps(report, indent=2))
            
            return report
    
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    
    # Run benchmark
    monitor.benchmark_processing(num_images=20)


def main():
    """Run all examples."""
    print("🔍 Advanced Face Recognition Software - Usage Examples")
    print("=" * 60)
    
    try:
        # Run all examples
        example_basic_usage()
        example_attendance_system()
        example_security_system()
        example_realtime_webcam()
        example_photo_organization()
        example_external_integration()
        example_performance_monitoring()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\nNext Steps:")
        print("1. Install the software: python setup.py install")
        print("2. Try the GUI: python face_recognition_system.py --gui")
        print("3. Start with webcam: python face_recognition_system.py --webcam")
        print("4. Register faces: python face_recognition_system.py --register 'Name' image.jpg")
        print("5. Run batch processing: python batch_processing.py --help")
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        print("  python setup.py install")
    
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("This is likely due to missing dependencies or system configuration.")


if __name__ == "__main__":
    main()
                