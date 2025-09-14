#!/usr/bin/env python3
"""
Batch Processing Utility for Face Recognition System
Processes multiple images or videos in batch mode
"""

import os
import cv2
import json
import argparse
from pathlib import Path
from typing import List, Dict
import time
from face_recognition_system import FaceRecognitionEngine

class BatchProcessor:
    """Batch processing utility for multiple files."""
    
    def __init__(self, tolerance: float = 0.6):
        self.engine = FaceRecognitionEngine(tolerance)
        self.results = []
    
    def process_image_directory(self, directory: str, output_dir: str = None) -> List[Dict]:
        """Process all images in a directory."""
        directory = Path(directory)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        results = []
        
        for image_path in directory.iterdir():
            if image_path.suffix.lower() in image_extensions:
                print(f"Processing: {image_path.name}")
                
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"Error loading {image_path.name}")
                    continue
                
                start_time = time.time()
                processed_image = self.engine.process_frame(image)
                processing_time = time.time() - start_time
                
                # Get recognition results
                locations, encodings = self.engine.detect_faces(image)
                faces_detected = []
                
                for encoding in encodings:
                    name, confidence = self.engine.identify_face(encoding)
                    faces_detected.append({
                        'name': name,
                        'confidence': confidence
                    })
                
                result = {
                    'file': image_path.name,
                    'faces_count': len(faces_detected),
                    'faces': faces_detected,
                    'processing_time': processing_time
                }
                results.append(result)
                
                # Save processed image if output directory specified
                if output_dir:
                    output_path = output_dir / f"processed_{image_path.name}"
                    cv2.imwrite(str(output_path), processed_image)
        
        return results
    
    def process_video_batch(self, video_paths: List[str], output_dir: str = None) -> List[Dict]:
        """Process multiple video files."""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        results = []
        
        for video_path in video_paths:
            print(f"Processing video: {video_path}")
            video_path = Path(video_path)
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Error opening {video_path}")
                continue
            
            # Video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Output video setup
            if output_dir:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_path = output_dir / f"processed_{video_path.name}"
                out = cv2.VideoWriter(str(output_path), fourcc, fps, 
                                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            
            faces_timeline = []
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 30th frame for efficiency
                if frame_number % 30 == 0:
                    locations, encodings = self.engine.detect_faces(frame)
                    frame_faces = []
                    
                    for encoding in encodings:
                        name, confidence = self.engine.identify_face(encoding)
                        frame_faces.append({
                            'name': name,
                            'confidence': confidence,
                            'timestamp': frame_number / fps
                        })
                    
                    faces_timeline.append({
                        'frame': frame_number,
                        'timestamp': frame_number / fps,
                        'faces': frame_faces
                    })
                
                processed_frame = self.engine.process_frame(frame)
                
                if output_dir:
                    out.write(processed_frame)
                
                frame_number += 1
                
                # Progress indicator
                if frame_number % 100 == 0:
                    progress = (frame_number / frame_count) * 100
                    print(f"Progress: {progress:.1f}%")
            
            cap.release()
            if output_dir:
                out.release()
            
            result = {
                'file': video_path.name,
                'total_frames': frame_count,
                'fps': fps,
                'duration': frame_count / fps,
                'faces_timeline': faces_timeline,
                'unique_faces': self._extract_unique_faces(faces_timeline)
            }
            results.append(result)
        
        return results
    
    def _extract_unique_faces(self, timeline: List[Dict]) -> List[str]:
        """Extract unique face names from timeline."""
        unique_faces = set()
        for frame_data in timeline:
            for face in frame_data['faces']:
                if face['name'] != 'Unknown':
                    unique_faces.add(face['name'])
        return list(unique_faces)
    
    def register_faces_from_directory(self, directory: str) -> Dict:
        """Register faces from directory with filename as name."""
        directory = Path(directory)
        results = {'success': [], 'failed': []}
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for image_path in directory.iterdir():
            if image_path.suffix.lower() in image_extensions:
                # Use filename (without extension) as person name
                name = image_path.stem
                
                image = cv2.imread(str(image_path))
                if image is None:
                    results['failed'].append(f"{name}: Could not load image")
                    continue
                
                if self.engine.register_face(image, name):
                    results['success'].append(name)
                    print(f"Registered: {name}")
                else:
                    results['failed'].append(f"{name}: Face registration failed")
                    print(f"Failed to register: {name}")
        
        return results
    
    def generate_report(self, results: List[Dict], output_file: str = "batch_report.json"):
        """Generate comprehensive report from batch processing results."""
        report = {
            'summary': {
                'total_files': len(results),
                'total_faces_detected': sum(r.get('faces_count', 0) for r in results),
                'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'average_processing_time': sum(r.get('processing_time', 0) for r in results) / len(results) if results else 0
            },
            'results': results,
            'statistics': self._calculate_statistics(results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: {output_file}")
        return report
    
    def _calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate processing statistics."""
        all_faces = []
        for result in results:
            if 'faces' in result:
                all_faces.extend(result['faces'])
        
        if not all_faces:
            return {}
        
        # Face recognition statistics
        recognized_faces = [f for f in all_faces if f['name'] != 'Unknown']
        unknown_faces = [f for f in all_faces if f['name'] == 'Unknown']
        
        # Confidence statistics
        confidences = [f['confidence'] for f in recognized_faces if f['confidence'] > 0]
        
        stats = {
            'total_faces': len(all_faces),
            'recognized_faces': len(recognized_faces),
            'unknown_faces': len(unknown_faces),
            'recognition_rate': len(recognized_faces) / len(all_faces) * 100 if all_faces else 0,
            'average_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'min_confidence': min(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0
        }
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Batch Face Recognition Processor")
    parser.add_argument("--images", type=str, help="Directory containing images to process")
    parser.add_argument("--videos", nargs='+', help="Video files to process")
    parser.add_argument("--register", type=str, help="Directory of images to register as faces")
    parser.add_argument("--output", type=str, help="Output directory for processed files")
    parser.add_argument("--tolerance", type=float, default=0.6, help="Recognition tolerance")
    parser.add_argument("--report", type=str, default="batch_report.json", help="Report output file")
    
    args = parser.parse_args()
    
    processor = BatchProcessor(tolerance=args.tolerance)
    results = []
    
    if args.register:
        print(f"Registering faces from: {args.register}")
        registration_results = processor.register_faces_from_directory(args.register)
        print(f"Successfully registered: {len(registration_results['success'])} faces")
        print(f"Failed to register: {len(registration_results['failed'])} faces")
        
        if registration_results['failed']:
            print("Failed registrations:")
            for failure in registration_results['failed']:
                print(f"  - {failure}")
    
    if args.images:
        print(f"Processing images from: {args.images}")
        image_results = processor.process_image_directory(args.images, args.output)
        results.extend(image_results)
        print(f"Processed {len(image_results)} images")
    
    if args.videos:
        print(f"Processing {len(args.videos)} videos")
        video_results = processor.process_video_batch(args.videos, args.output)
        results.extend(video_results)
        print(f"Processed {len(video_results)} videos")
    
    if results:
        report = processor.generate_report(results, args.report)
        
        # Print summary
        print("\n=== PROCESSING SUMMARY ===")
        print(f"Total files processed: {report['summary']['total_files']}")
        print(f"Total faces detected: {report['summary']['total_faces_detected']}")
        print(f"Average processing time: {report['summary']['average_processing_time']:.2f}s")
        
        if 'statistics' in report and report['statistics']:
            stats = report['statistics']
            print(f"Recognition rate: {stats['recognition_rate']:.1f}%")
            print(f"Average confidence: {stats['average_confidence']:.1f}%")
    
    if not any([args.images, args.videos, args.register]):
        print("No processing action specified. Use --help for options.")


if __name__ == "__main__":
    main()
