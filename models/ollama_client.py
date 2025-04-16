import requests
import json

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate"
    
    def generate(self, model, prompt, system_prompt=None):
        """
        Generate a response from an Ollama model.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(self.api_endpoint, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error communicating with Ollama: {str(e)}")
    
    def extract_cv_data(self, cv_text, model):
        """
        Extract structured data from CV text using the specified model.
        """
        system_prompt = """
        You are an expert CV/resume parser. Your task is to extract structured information from the CV text provided.
        Extract the following fields in JSON format:
        - name: The full name of the candidate
        - email: Email address
        - phone: Phone number
        - education: List of educational qualifications with institution, degree, field, and years
        - skills: List of technical and soft skills
        - experience: List of work experiences with company, position, duration, and key responsibilities
        {
  "name": ,
  "email": ,
  "phone":,
  "education": [
    {
      "degree": ,
      "field":, 
      "institution": ,
      "years": 
    }
  ],
  "experience": [
    {
      "position":,
      "company": ,
      "duration": ,
      "key_responsibilities": []
    },
  
  ],
  "skills": list of skills (str) [skill1,skill2 .....]}
        Return ONLY valid JSON with these fields and no additional explanation or text.
        """
        
        prompt = f"""
        Please extract the structured information from the following CV:
        
        {cv_text}
        
        Extract the data in JSON format with the fields: name, email, phone, education, skills, and experience.
        """
        
        try:
            response = self.generate(model, prompt, system_prompt)
            
            # Try to parse the response as JSON
            try:
                # Find JSON content if it's embedded in explanatory text
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = response[json_start:json_end]
                    extracted_data = json.loads(json_content)
                else:
                    # If no JSON-like structure is found, use the whole response
                    extracted_data = json.loads(response)
                
                return extracted_data
            except json.JSONDecodeError:
                # If parsing fails, return a structured error message
                return {
                    "error": "Failed to parse model output as JSON",
                    "raw_response": response
                }
                
        except Exception as e:
            return {
                "error": f"Error during extraction: {str(e)}",
                "raw_response": ""
            }