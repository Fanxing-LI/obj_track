#!/usr/bin/env python3

import rosbag
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import os
import sys
import argparse
from datetime import datetime
import yaml

class RosbagVideoExporter:
    """
    A comprehensive ROS bag to video exporter with multiple format support
    """
    
    def __init__(self):
        self.bridge = CvBridge()
        self.supported_image_types = ['sensor_msgs/Image', 'sensor_msgs/CompressedImage']
        
    def list_topics(self, bag_file_path, show_all=False):
        """
        List topics in the ROS bag file, with option to filter image topics
        
        Args:
            bag_file_path: Path to the ROS bag file
            show_all: If True, show all topics; if False, only show image topics
            
        Returns:
            dict: Dictionary of topic information
        """
        if not os.path.exists(bag_file_path):
            raise FileNotFoundError(f"Bag file '{bag_file_path}' not found!")
        
        topics_info = {}
        
        with rosbag.Bag(bag_file_path, 'r') as bag:
            info = bag.get_type_and_topic_info()
            all_topics = info[1]
            
            for topic_name, topic_info in all_topics.items():
                msg_type = topic_info.msg_type
                msg_count = topic_info.message_count
                
                # Filter for image topics if requested
                if not show_all:
                    if msg_type not in self.supported_image_types:
                        continue
                
                topics_info[topic_name] = {
                    'type': msg_type,
                    'count': msg_count,
                    'is_image': msg_type in self.supported_image_types
                }
        
        return topics_info
    
    def get_bag_time_range(self, bag_file_path):
        """
        Get the time range of the bag file
        
        Args:
            bag_file_path: Path to the ROS bag file
            
        Returns:
            tuple: (start_time, end_time) in seconds
        """
        with rosbag.Bag(bag_file_path, 'r') as bag:
            start_time = bag.get_start_time()
            end_time = bag.get_end_time()
            return start_time, end_time
    
    def export_video(self, bag_file_path, topic_name, output_path, 
                    fps=30, start_time=None, end_time=None, 
                    resize=None, quality='high', codec='mp4v'):
        """
        Export video from ROS bag
        
        Args:
            bag_file_path: Path to the ROS bag file
            topic_name: Name of the image topic
            output_path: Output video file path
            fps: Output video FPS
            start_time: Start time in seconds (None for beginning)
            end_time: End time in seconds (None for end)
            resize: Tuple (width, height) for resizing (None for original)
            quality: Video quality ('high', 'medium', 'low')
            codec: Video codec ('mp4v', 'XVID', 'H264')
            
        Returns:
            bool: Success status
        """
        if not os.path.exists(bag_file_path):
            raise FileNotFoundError(f"Bag file '{bag_file_path}' not found!")
        
        # Set quality parameters
        quality_params = {
            'high': {'bitrate': -1, 'crf': 18},
            'medium': {'bitrate': 2000000, 'crf': 23},
            'low': {'bitrate': 1000000, 'crf': 28}
        }
        
        video_writer = None
        frame_count = 0
        processed_count = 0
        
        print(f"Exporting video from: {bag_file_path}")
        print(f"Topic: {topic_name}")
        print(f"Output: {output_path}")
        
        with rosbag.Bag(bag_file_path, 'r') as bag:
            # Verify topic exists
            topics = bag.get_type_and_topic_info()[1]
            if topic_name not in topics:
                available = [t for t, info in topics.items() 
                           if info.msg_type in self.supported_image_types]
                raise ValueError(f"Topic '{topic_name}' not found! "
                               f"Available image topics: {available}")
            
            topic_info = topics[topic_name]
            total_messages = topic_info.message_count
            print(f"Total messages: {total_messages}")
            
            # Get time range if specified
            bag_start, bag_end = self.get_bag_time_range(bag_file_path)
            if start_time is None:
                start_time = bag_start
            if end_time is None:
                end_time = bag_end
                
            print(f"Time range: {start_time:.2f} - {end_time:.2f} seconds")
            
            # Process messages
            for topic, msg, timestamp in bag.read_messages(topics=[topic_name]):
                msg_time = timestamp.to_sec()
                
                # Skip messages outside time range
                if msg_time < start_time or msg_time > end_time:
                    continue
                
                try:
                    # Convert message to OpenCV image
                    if msg._type == 'sensor_msgs/Image':
                        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    elif msg._type == 'sensor_msgs/CompressedImage':
                        np_arr = np.frombuffer(msg.data, np.uint8)
                        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    else:
                        print(f"Unsupported message type: {msg._type}")
                        continue
                    
                    if cv_image is None:
                        print(f"Failed to decode image at frame {frame_count}")
                        continue
                    
                    # Resize if requested
                    if resize is not None:
                        cv_image = cv2.resize(cv_image, resize)
                    
                    # Initialize video writer
                    if video_writer is None:
                        height, width = cv_image.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        
                        # Create output directory if needed
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                        if not video_writer.isOpened():
                            raise RuntimeError("Failed to open video writer")
                        
                        print(f"Video dimensions: {width}x{height}")
                        print(f"Codec: {codec}, FPS: {fps}")
                    
                    # Write frame
                    video_writer.write(cv_image)
                    processed_count += 1
                    
                    # Progress indicator
                    if processed_count % 50 == 0:
                        progress = (msg_time - start_time) / (end_time - start_time) * 100
                        print(f"Progress: {progress:.1f}% ({processed_count} frames)")
                        
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                
                frame_count += 1
        
        # Cleanup
        if video_writer is not None:
            video_writer.release()
            
        if processed_count > 0:
            print(f"✓ Video exported successfully!")
            print(f"  Frames processed: {processed_count}")
            print(f"  Output file: {output_path}")
            return True
        else:
            print("✗ No frames were processed!")
            return False
    
    def export_multiple_topics(self, bag_file_path, topic_configs, output_dir):
        """
        Export multiple topics to separate videos
        
        Args:
            bag_file_path: Path to the ROS bag file
            topic_configs: List of dicts with topic configurations
            output_dir: Output directory for videos
            
        Returns:
            dict: Results for each topic
        """
        results = {}
        
        for config in topic_configs:
            topic_name = config['topic']
            output_name = config.get('output_name', topic_name.replace('/', '_') + '.mp4')
            output_path = os.path.join(output_dir, output_name)
            
            print(f"\n{'='*60}")
            print(f"Processing topic: {topic_name}")
            print(f"{'='*60}")
            
            try:
                success = self.export_video(
                    bag_file_path=bag_file_path,
                    topic_name=topic_name,
                    output_path=output_path,
                    fps=config.get('fps', 30),
                    start_time=config.get('start_time'),
                    end_time=config.get('end_time'),
                    resize=config.get('resize'),
                    quality=config.get('quality', 'high'),
                    codec=config.get('codec', 'mp4v')
                )
                results[topic_name] = {'success': success, 'output_path': output_path}
            except Exception as e:
                print(f"✗ Failed to process {topic_name}: {e}")
                results[topic_name] = {'success': False, 'error': str(e)}
        
        return results

def create_config_template(output_path):
    """
    Create a configuration template file for batch processing
    """
    config = {
        'bag_file': 'path/to/your/bagfile.bag',
        'output_dir': 'output_videos/',
        'topics': [
            {
                'topic': '/camera/color/image_raw',
                'output_name': 'rgb_camera.mp4',
                'fps': 30,
                'start_time': None,  # Start from beginning
                'end_time': None,    # Process until end
                'resize': None,      # Keep original size, or [width, height]
                'quality': 'high',   # 'high', 'medium', 'low'
                'codec': 'mp4v'      # 'mp4v', 'XVID', 'H264'
            },
            {
                'topic': '/camera/depth/image_raw',
                'output_name': 'depth_camera.mp4',
                'fps': 30,
                'resize': [640, 480],
                'quality': 'medium'
            }
        ]
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration template created: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Export video from ROS bag files')
    parser.add_argument('bag_file', help='Path to ROS bag file')
    parser.add_argument('--topic', '-t', help='Image topic name')
    parser.add_argument('--output', '-o', help='Output video file path')
    parser.add_argument('--fps', type=int, default=30, help='Output video FPS')
    parser.add_argument('--start', type=float, help='Start time in seconds')
    parser.add_argument('--end', type=float, help='End time in seconds')
    parser.add_argument('--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                       help='Resize video to specified dimensions')
    parser.add_argument('--quality', choices=['high', 'medium', 'low'], default='high',
                       help='Video quality')
    parser.add_argument('--codec', default='mp4v', help='Video codec')
    parser.add_argument('--list-topics', action='store_true',
                       help='List all image topics in the bag file')
    parser.add_argument('--config', help='Use YAML config file for batch processing')
    parser.add_argument('--create-config', help='Create configuration template file')
    
    args = parser.parse_args()
    
    exporter = RosbagVideoExporter()
    
    # Create config template
    if args.create_config:
        create_config_template(args.create_config)
        return
    
    # Check if bag file exists
    if not os.path.exists(args.bag_file):
        print(f"Error: Bag file '{args.bag_file}' not found!")
        sys.exit(1)
    
    # List topics mode
    if args.list_topics:
        print("Image topics in bag file:")
        print("="*50)
        try:
            topics = exporter.list_topics(args.bag_file, show_all=False)
            if not topics:
                print("No image topics found!")
                # Show all topics as fallback
                print("\nAll topics:")
                all_topics = exporter.list_topics(args.bag_file, show_all=True)
                for topic, info in all_topics.items():
                    print(f"  {topic} ({info['type']}) - {info['count']} messages")
            else:
                for topic, info in topics.items():
                    print(f"  {topic} ({info['type']}) - {info['count']} messages")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        return
    
    # Batch processing with config file
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            bag_file = config['bag_file']
            output_dir = config['output_dir']
            topic_configs = config['topics']
            
            os.makedirs(output_dir, exist_ok=True)
            
            results = exporter.export_multiple_topics(bag_file, topic_configs, output_dir)
            
            print(f"\n{'='*60}")
            print("BATCH PROCESSING SUMMARY")
            print(f"{'='*60}")
            for topic, result in results.items():
                status = "✓" if result['success'] else "✗"
                print(f"{status} {topic}")
                if not result['success'] and 'error' in result:
                    print(f"    Error: {result['error']}")
                elif result['success']:
                    print(f"    Output: {result['output_path']}")
            
        except Exception as e:
            print(f"Error processing config file: {e}")
            sys.exit(1)
        return
    
    # Single topic export mode
    if not args.topic:
        print("Error: --topic is required for single export mode")
        print("Use --list-topics to see available topics")
        sys.exit(1)
    
    if not args.output:
        # Auto-generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_name = args.topic.replace('/', '_')
        args.output = f"video_{topic_name}_{timestamp}.mp4"
    
    try:
        success = exporter.export_video(
            bag_file_path=args.bag_file,
            topic_name=args.topic,
            output_path=args.output,
            fps=args.fps,
            start_time=args.start,
            end_time=args.end,
            resize=tuple(args.resize) if args.resize else None,
            quality=args.quality,
            codec=args.codec
        )
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
