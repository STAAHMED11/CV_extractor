import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import uuid
from processors.pdf_processor import PDFProcessor
from processors.ocr_processor import OCRProcessor
from models.ollama_client import OllamaClient
from evaluator.metrics import calculate_metrics
import json

app = Flask(__name__)
app.secret_key = "cv-extractor-secret-key"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize processors and models
pdf_processor = PDFProcessor()
ocr_processor = OCRProcessor()
ollama_client = OllamaClient()

ALLOWED_EXTENSIONS = {'pdf','png','jpg','JPEG','jpeg'}
MODELS = ['llama3.2', 'gemma3:1b', 'phi3']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    models = MODELS
    return render_template('index.html', models=models)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    model = request.form.get('model', MODELS[0])
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        unique_filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Process the file
        try:
            # Determine if the PDF is text-based or image-based
            is_text_pdf = pdf_processor.is_text_based(file_path)
            
            if is_text_pdf:
                text_content = pdf_processor.extract_text(file_path)
                cv_type = "text-based"
            else:
                text_content = ocr_processor.process_scanned_pdf(file_path)
                cv_type = "image-based (OCR processed)"
            
            # Process with selected LLM
            extracted_data = ollama_client.extract_cv_data(text_content, model)
            
            # Save results
            result_id = uuid.uuid4().hex
            result_path = os.path.join('results', f"{result_id}.json")
            os.makedirs('results', exist_ok=True)
            
            result = {
                "file_name": file.filename,
                "cv_type": cv_type,
                "model_used": model,
                "extracted_data": extracted_data
            }
            
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            return redirect(url_for('show_result', result_id=result_id))
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a PDF file.')
    return redirect(url_for('index'))

@app.route('/result/<result_id>')
def show_result(result_id):
    try:
        with open(os.path.join('results', f"{result_id}.json"), 'r') as f:
            result = json.load(f)
        return render_template('result.html', result=result)
    except FileNotFoundError:
        flash('Result not found')
        return redirect(url_for('index'))

@app.route('/compare', methods=['GET', 'POST'])
def compare_models():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            unique_filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Process with all models
            try:
                # Determine if the PDF is text-based or image-based
                is_text_pdf = pdf_processor.is_text_based(file_path)
                
                if is_text_pdf:
                    text_content = pdf_processor.extract_text(file_path)
                    cv_type = "text-based"
                else:
                    text_content = ocr_processor.process_scanned_pdf(file_path)
                    cv_type = "image-based (OCR processed)"
                
                # Process with all models
                results = {}
                for model in MODELS:
                    extracted_data = ollama_client.extract_cv_data(text_content, model)
                    results[model] = extracted_data
                
                # Save comparison results
                comparison_id = uuid.uuid4().hex
                comparison_path = os.path.join('comparisons', f"{comparison_id}.json")
                os.makedirs('comparisons', exist_ok=True)
                
                comparison = {
                    "file_name": file.filename,
                    "cv_type": cv_type,
                    "results": results
                }
                
                with open(comparison_path, 'w') as f:
                    json.dump(comparison, f, indent=2)
                
                return redirect(url_for('show_comparison', comparison_id=comparison_id))
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return redirect(url_for('compare'))
        
        flash('Invalid file type. Please upload a PDF file.')
        return redirect(url_for('compare'))
    
    return render_template('compare.html', models=MODELS)

@app.route('/comparison/<comparison_id>')
def show_comparison(comparison_id):
    try:
        with open(os.path.join('comparisons', f"{comparison_id}.json"), 'r') as f:
            comparison = json.load(f)
        return render_template('comparison.html', comparison=comparison)
    except FileNotFoundError:
        flash('Comparison not found')
        return redirect(url_for('compare'))

@app.route('/evaluate')
def evaluate():
    # For the evaluation page where users can see model performance metrics
    metrics = calculate_metrics()  # This would compare against ground truth
    print(json.dumps(metrics, indent=2))  # Debug print
    # Calculate the maximum F1 score across all models
    max_f1_score = 0
    for model in metrics['models']:
        if metrics['summary'][model]['f1_score'] > max_f1_score:
            max_f1_score = metrics['summary'][model]['f1_score']
    
    return render_template('evaluation.html', metrics=metrics, max_f1_score=max_f1_score)

if __name__ == '__main__':
    app.run(debug=True)