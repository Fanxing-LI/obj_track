#!/usr/bin/env python3

import rosbag
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import sys

# Define parameters directly in the file
RGB_TOPIC_NAME = "/camera/color/image_raw"  # Change this to your RGB topic name
BAG_FILE_PATH = "plots/lfx_tracking_91_2.bag"  # Change this to your bag file path
OUTPUT_VIDEO_PATH = "first_view.mp4"      # Change this to your desired output path

RGB_TOPIC_NAME = "/stabilized/image_rect"  # Change this to your RGB topic name
BAG_FILE_PATH = "plots/stabilized_output.bag"  # Change this to your bag file path
OUTPUT_VIDEO_PATH = "stabilized_output.mp4"      # Change this to your desired output path
VIDEO_FPS = 30                              # Change this to your desired FPS

def list_bag_topics(bag_file_path):
    """
    List all topics in the ROS bag file
    
    Args:
        bag_file_path: Path to the ROS bag file
    """
    
    # Check if bag file exists
    if not os.path.exists(bag_file_path):
        print(f"Error: Bag file '{bag_file_path}' not found!")
        return False
    
    print(f"Loading ROS bag file: {bag_file_path}")
    
    try:
        # Open the bag file
        with rosbag.Bag(bag_file_path, 'r') as bag:
            
            # Get bag info
            info = bag.get_type_and_topic_info()
            topics_info = info[1]  # topic info dictionary
            
            print("=" * 80)
            print("ALL TOPICS IN BAG FILE:")
            print("=" * 80)
            print(f"{'Topic Name':<50} {'Message Type':<25} {'Count':<10}")
            print("-" * 80)
            
            # Sort topics by name for better readability
            for topic_name in sorted(topics_info.keys()):
                topic_info = topics_info[topic_name]
                msg_type = topic_info.msg_type
                msg_count = topic_info.message_count
                
                print(f"{topic_name:<50} {msg_type:<25} {msg_count:<10}")
                
                # Highlight image topics
                if 'Image' in msg_type or 'image' in topic_name.lower():
                    print(f"  → This looks like an IMAGE topic! ←")
            
            print("-" * 80)
            print(f"Total topics: {len(topics_info)}")
            print("=" * 80)
            
            # Show only image-related topics
            image_topics = []
            for topic_name, topic_info in topics_info.items():
                if 'Image' in topic_info.msg_type or 'image' in topic_name.lower():
                    image_topics.append((topic_name, topic_info.msg_type, topic_info.message_count))
            
            if image_topics:
                print("\nIMAGE TOPICS FOUND:")
                print("=" * 60)
                for topic_name, msg_type, msg_count in image_topics:
                    print(f"Topic: {topic_name}")
                    print(f"  Type: {msg_type}")
                    print(f"  Messages: {msg_count}")
                    print("-" * 40)
            else:
                print("\nNo obvious image topics found.")
                print("Look for topics that might contain images above.")
            
            return True
                
    except Exception as e:
        print(f"Error reading bag file: {e}")
        return False

def bag_to_video(bag_file_path, rgb_topic_name, output_video_path, fps=30):
    """
    Extract RGB images from ROS bag file and convert to MP4 video
    
    Args:
        bag_file_path: Path to the ROS bag file
        rgb_topic_name: Name of the RGB image topic
        output_video_path: Path for output MP4 file
        fps: Frames per second for output video
    """
    
    # Initialize CV bridge for image conversion
    bridge = CvBridge()
    
    # Check if bag file exists
    if not os.path.exists(bag_file_path):
        print(f"Error: Bag file '{bag_file_path}' not found!")
        return False
    
    print(f"Loading ROS bag file: {bag_file_path}")
    
    try:
        # Open the bag file
        with rosbag.Bag(bag_file_path, 'r') as bag:
            
            # Check if the topic exists in the bag
            topics = bag.get_type_and_topic_info()[1].keys()
            if rgb_topic_name not in topics:
                print(f"Error: Topic '{rgb_topic_name}' not found in bag file!")
                print(f"Available topics: {list(topics)}")
                return False
            
            # Get bag info
            info = bag.get_type_and_topic_info()
            topic_info = info[1][rgb_topic_name]
            message_count = topic_info.message_count
            
            print(f"Found topic: {rgb_topic_name}")
            print(f"Message count: {message_count}")
            print(f"Message type: {topic_info.msg_type}")
            
            if message_count == 0:
                print("No messages found in the topic!")
                return False
            
            # Initialize video writer (will be set after first frame)
            video_writer = None
            frame_count = 0
            
            print("Processing images...")
            
            # Read messages from the specified topic
            for topic, msg, timestamp in bag.read_messages(topics=[rgb_topic_name]):
                try:
                    # Convert ROS image message to OpenCV format
                    if msg._type == 'sensor_msgs/Image':
                        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    else:
                        print(f"Unsupported message type: {msg._type}")
                        continue
                    
                    # Initialize video writer with first frame dimensions
                    if video_writer is None:
                        height, width = cv_image.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                        print(f"Video dimensions: {width}x{height}")
                        print(f"Output FPS: {fps}")
                    
                    # Write frame to video
                    video_writer.write(cv_image)
                    frame_count += 1
                    
                    # Progress indicator
                    if frame_count % 100 == 0:
                        print(f"Processed {frame_count}/{message_count} frames...")
                        
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    continue
            
            # Clean up
            if video_writer is not None:
                video_writer.release()
                print(f"Video saved: {output_video_path}")
                print(f"Total frames processed: {frame_count}")
                return True
            else:
                print("No valid frames found!")
                return False
                
    except Exception as e:
        print(f"Error reading bag file: {e}")
        return False

def main():
    """
    Main function using predefined parameters
    """
    print("=" * 60)
    print("ROS Bag to MP4 Video Converter")
    print("=" * 60)
    
    # First, list all topics in the bag file
    print("Step 1: Listing all topics in the bag file...")
    success = list_bag_topics(BAG_FILE_PATH)
    
    if not success:
        print("✗ Failed to read bag file!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("CURRENT CONVERSION SETTINGS:")
    print("=" * 60)
    print(f"Input bag file: {BAG_FILE_PATH}")
    print(f"RGB topic name: {RGB_TOPIC_NAME}")
    print(f"Output video: {OUTPUT_VIDEO_PATH}")
    print(f"Target FPS: {VIDEO_FPS}")
    print("=" * 60)
    
    # Ask user if they want to proceed with current settings
    print("\nStep 2: Video Conversion")
    print("If you want to use a different topic, please modify RGB_TOPIC_NAME in the script.")
    
    user_input = input("\nProceed with video conversion? (y/n): ").lower().strip()
    
    if user_input != 'y' and user_input != 'yes':
        print("Conversion cancelled. Please modify RGB_TOPIC_NAME and run again.")
        return
    
    # Convert bag to video
    print("\nStarting video conversion...")
    success = bag_to_video(BAG_FILE_PATH, RGB_TOPIC_NAME, OUTPUT_VIDEO_PATH, VIDEO_FPS)
    
    if success:
        print("✓ Conversion completed successfully!")
    else:
        print("✗ Conversion failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()