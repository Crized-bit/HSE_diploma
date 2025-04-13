#!/usr/bin/env python3
import os
import json
from collections import defaultdict

def find_unique_labels(annotations_dir):
    """
    Scans all JSON files in the specified directory and extracts unique labels.
    
    Args:
        annotations_dir (str): Path to the directory containing annotation files
        
    Returns:
        dict: A dictionary where keys are unique labels and values are counts
    """
    # Dictionary to store labels and their counts
    all_labels = defaultdict(int)
    
    # Walk through all files in the annotations directory
    for root, _, files in os.walk(annotations_dir):
        for filename in files:
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                
                try:
                    # Load the JSON file
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract labels from each entity
                    if 'entities' in data:
                        for entity in data['entities']:
                            if 'labels' in entity:
                                for label in entity['labels'].keys():
                                    all_labels[label] += 1
                
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    
    return all_labels

if __name__ == "__main__":
    # Path to the annotations directory
    annotations_dir = "/home/jovyan/p.kudrevatyh/yolov5_motion/data/annotations"
    
    # Get unique labels
    unique_labels = find_unique_labels(annotations_dir)
    
    # Print results
    print("\nUnique labels found in annotations:")
    print("================================")
    for label, count in sorted(unique_labels.items()):
        print(f"{label}: {count} occurrences")
    
    print(f"\nTotal unique labels: {len(unique_labels)}")