import json
import os
import re
from difflib import SequenceMatcher

def load_ground_truth(ground_truth_path):
    """
    Load ground truth data from a JSON file.
    """
    with open(ground_truth_path, 'r') as f:
        return json.load(f)

def normalize_text(text):
    """
    Normalize text for comparison by removing extra spaces, lowercasing, etc.
    """
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
    """
    Calculate similarity between extracted field and ground truth field.
    """
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

def calculate_field_metrics(extracted_data, ground_truth, field_name):
    """
    Calculate precision, recall, and F1 score for a specific field.
    """
    if field_name not in ground_truth:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "similarity": 0.0
        }
    
    # If field is missing in extracted data
    if field_name not in extracted_data:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "similarity": 0.0
        }
    
    # Calculate similarity between extracted and ground truth
    similarity = field_similarity(extracted_data[field_name], ground_truth[field_name])
    
    # For simplicity, using similarity as both precision and recall
    # In a real scenario, you might have more complex calculations
    precision = similarity
    recall = similarity
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
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
    """
    Calculate metrics for all CVs and models in the evaluation dataset.
    """
    # Directory containing ground truth data
    ground_truth_dir = "data/ground_truth"
    # Directory containing model outputs
    results_dir = "data/results"
    
    if not os.path.exists(ground_truth_dir) or not os.path.exists(results_dir):
        return {
            "error": "Evaluation data not found",
            "models": [],
            "metrics": {}
        }
    
    models = ["llama3", "mistral", "phi"]
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