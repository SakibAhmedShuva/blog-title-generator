# app.py
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Add CORS support with more explicit settings
CORS(app, resources={r"/*": {"origins": "*"}})

# Print debug information
print(f"Current working directory: {os.getcwd()}")
template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
print(f"Template folder path: {template_path}")
index_path = os.path.join(template_path, "index.html")
print(f"Index.html path: {index_path}")
print(f"Index.html exists: {os.path.exists(index_path)}")

# Initialize variables
title_chain = None
llm = None

# Initialize Groq LLM
try:
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL_NAME", "llama3-70b-8192")
    print(f"API Key found: {'Yes' if api_key else 'No'}")
    print(f"Using model: {model_name}")
    
    llm = ChatGroq(
        api_key=api_key,
        model_name=model_name,
    )
    
    # Create prompt template for generating blog titles
    title_template = PromptTemplate(
        input_variables=["topic", "tone", "instructions"],
        template="""Generate 5 creative and engaging blog titles about {topic}.
        
        The titles should have a {tone} tone.
        
        Additional instructions: {instructions}
        
        Return only the titles in a numbered list format."""
    )

    # Create chain
    title_chain = LLMChain(llm=llm, prompt=title_template)
    print("LLM chain initialized successfully")
    
except Exception as e:
    print(f"Error initializing LLM: {str(e)}")

@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        return f"Error rendering template: {str(e)}", 500

@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory("static", path)

@app.route("/generate", methods=["POST"])
def generate_titles():
    try:
        # Check if title_chain is initialized
        if title_chain is None:
            return jsonify({"error": "LLM not initialized properly. Check your API key and model configuration."}), 500
            
        # Print debug info about the request
        print(f"Request received: {request.method}")
        print(f"Content-Type: {request.headers.get('Content-Type')}")
        
        # Get data from request
        data = request.json
        print(f"Request data: {data}")
        
        topic = data.get("topic", "")
        tone = data.get("tone", "casual")
        instructions = data.get("instructions", "Make the titles catchy and SEO-friendly")
        
        # Validate input
        if not topic:
            return jsonify({"error": "Blog topic is required"}), 400
            
        # Generate titles
        print(f"Generating titles for topic: {topic}, tone: {tone}")
        result = title_chain.run(topic=topic, tone=tone, instructions=instructions)
        print(f"Generated result: {result}")
        
        # Parse the result into a list of titles
        titles = [title.strip() for title in result.split("\n") if title.strip()]
        print(f"Parsed titles: {titles}")
        
        return jsonify({"titles": titles})
    
    except Exception as e:
        print(f"Error in generate_titles: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/test")
def test():
    return jsonify({"status": "API is working"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)