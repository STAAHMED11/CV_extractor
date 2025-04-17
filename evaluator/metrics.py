import json
import os
import re
from difflib import SequenceMatcher

def load_ground_truth(ground_truth_path):
 
    with open(ground_truth_path, 'r') as f:
        return json.load(f)

def normalize_text(text):
  
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove punctuation for comparison
    text = re.sub(r'[^\w\s]', '', text)
    return text

def field_similarity(extracted_value, ground_truth_value):

    # Handle different data types
    if isinstance(extracted_value, list) and isinstance(ground_truth_value, list):
        # For lists (like skills), calculate average similarity of best matches
        if not extracted_value or not ground_truth_value:
            return 0.0
        
        total_sim = 0
        count = 0
        
        # For each ground truth item, find the best match in extracted items
        for gt_item in ground_truth_value:
            best_sim = 0
            for ext_item in extracted_value:
                sim = SequenceMatcher(None, 
                                      normalize_text(str(gt_item)), 
                                      normalize_text(str(ext_item))).ratio()
                best_sim = max(best_sim, sim)
            
            total_sim += best_sim
            count += 1
        
        return total_sim / count if count > 0 else 0.0
    
    elif isinstance(extracted_value, dict) and isinstance(ground_truth_value, dict):
        # For dictionaries (like nested structures), calculate average field similarity
        if not extracted_value or not ground_truth_value:
            return 0.0
        
        total_sim = 0
        count = 0
        
        for key in ground_truth_value:
            if key in extracted_value:
                sim = field_similarity(extracted_value[key], ground_truth_value[key])
                total_sim += sim
                count += 1
        
        return total_sim / count if count > 0 else 0.0
    
    else:
        # For simple fields (strings, numbers), use sequence matcher
        return SequenceMatcher(None, 
                             normalize_text(str(extracted_value)), 
                             normalize_text(str(ground_truth_value))).ratio()
import difflib

import difflib

def tokenize(text):
    """Ensure text is a string, then tokenize into lowercase words."""
    if isinstance(text, list):
        text = ' '.join(map(str, text))
    elif not isinstance(text, str):
        text = str(text)
    return set(text.strip().lower().split())

def calculate_field_metrics(extracted_data, ground_truth, field_name):
    """
    Calculate precision, recall, F1 score, and similarity for a specific field using token-based evaluation.
    """
    if field_name not in ground_truth or field_name not in extracted_data:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "similarity": 0.0
        }

    predicted = extracted_data[field_name]
    actual = ground_truth[field_name]

    # Tokenize both
    pred_tokens = tokenize(predicted)
    actual_tokens = tokenize(actual)

    # Compute precision, recall
    true_positives = len(pred_tokens & actual_tokens)
    false_positives = len(pred_tokens - actual_tokens)
    false_negatives = len(actual_tokens - pred_tokens)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Similarity
    pred_str = ' '.join(pred_tokens)
    actual_str = ' '.join(actual_tokens)
    similarity = difflib.SequenceMatcher(None, pred_str, actual_str).ratio()

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "similarity": similarity
    }


def calculate_cv_metrics(extracted_data, ground_truth):
    """
    Calculate metrics for all fields in a CV.
    """
    fields = ["name", "email", "phone", "education", "skills", "experience"]
    metrics = {}
    
    for field in fields:
        metrics[field] = calculate_field_metrics(extracted_data, ground_truth, field)
    
    # Calculate overall metrics (average across fields)
    overall_precision = sum(metrics[field]["precision"] for field in fields) / len(fields)
    overall_recall = sum(metrics[field]["recall"] for field in fields) / len(fields)
    overall_f1 = sum(metrics[field]["f1_score"] for field in fields) / len(fields)
    
    metrics["overall"] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1_score": overall_f1
    }
    
    return metrics

def calculate_metrics():
    # Directory containing ground truth data
    ground_truth_dir = "data/ground_truth"
    # Directory containing model outputs
    results_dir = "data/results"
    
    print(f"Ground truth directory exists: {os.path.exists(ground_truth_dir)}")
    print(f"Results directory exists: {os.path.exists(results_dir)}")
    
    # List all files
    if os.path.exists(ground_truth_dir):
        print(f"Ground truth files: {os.listdir(ground_truth_dir)}")
    if os.path.exists(results_dir):
        print(f"Result files: {os.listdir(results_dir)}")

    
    if not os.path.exists(ground_truth_dir) or not os.path.exists(results_dir):
        return {
            "error": "Evaluation data not found",
            "models": [],
            "metrics": {}
        }
    
    models = ["llama3.2", "mistral", "qwen2.5"]
    all_metrics = {model: {} for model in models}
    
    # List all ground truth files
    ground_truth_files = [f for f in os.listdir(ground_truth_dir) if f.endswith('.json')]
    for gt_file in ground_truth_files:
        cv_id = gt_file.split('.')[0]
        ground_truth_path = os.path.join(ground_truth_dir, gt_file)
        ground_truth = load_ground_truth(ground_truth_path)
        
        for model in models:
            result_file = f"{cv_id}_{model}.json"
            result_path = os.path.join(results_dir, result_file)
            
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    extracted_data = json.load(f)
                
                metrics = calculate_cv_metrics(extracted_data, ground_truth)
                all_metrics[model][cv_id] = metrics
    
    # Calculate average metrics across all CVs for each model
    summary = {model: {
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "field_metrics": {
            "name": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
            "email": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
            "phone": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
            "education": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
            "skills": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
            "experience": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
        }
    } for model in models}
    print(models)
    for model in models:
        cv_count = len(all_metrics[model])
        
        if cv_count > 0:
            # Calculate average overall metrics
            overall_precision = sum(all_metrics[model][cv_id]["overall"]["precision"] for cv_id in all_metrics[model]) / cv_count
            overall_recall = sum(all_metrics[model][cv_id]["overall"]["recall"] for cv_id in all_metrics[model]) / cv_count
            overall_f1 = sum(all_metrics[model][cv_id]["overall"]["f1_score"] for cv_id in all_metrics[model]) / cv_count
            
            summary[model]["precision"] = overall_precision
            summary[model]["recall"] = overall_recall
            summary[model]["f1_score"] = overall_f1
            
            # Calculate average field metrics
            fields = ["name", "email", "phone", "education", "skills", "experience"]
            
            for field in fields:
                field_precision = sum(all_metrics[model][cv_id][field]["precision"] for cv_id in all_metrics[model]) / cv_count
                field_recall = sum(all_metrics[model][cv_id][field]["recall"] for cv_id in all_metrics[model]) / cv_count
                field_f1 = sum(all_metrics[model][cv_id][field]["f1_score"] for cv_id in all_metrics[model]) / cv_count
                
                summary[model]["field_metrics"][field]["precision"] = field_precision
                summary[model]["field_metrics"][field]["recall"] = field_recall
                summary[model]["field_metrics"][field]["f1_score"] = field_f1
    
    return {
        "models": models,
        "individual_metrics": all_metrics,
        "summary": summary
    }