import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import json
import re
import hashlib
from graphviz import Digraph
import os
from dotenv import load_dotenv

# --- CHANGE 1: Load environment variables from a .env file ---
load_dotenv()

def sanitize_filename(title):
    return re.sub(r'[^\w\-]', '', title.strip())

def hash_id(value):
    return hashlib.md5(value.encode()).hexdigest()[:8]

def add_node(dot, node_id, label, shape='box', style='rounded,filled', fillcolor='lightblue'):
    dot.node(node_id, label, shape=shape, style=style, fillcolor=fillcolor)

def build_flowchart(dot, data, parent_id=None, level=0):
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lavender', 'peachpuff', 'mistyrose', 'honeydew', 'thistle']
    current_color = colors[level % len(colors)]

    if isinstance(data, dict):
        for key, value in data.items():
            node_label = str(key).strip()
            key_id = hash_id(f"{parent_id}_{node_label}" if parent_id else node_label)
            add_node(dot, key_id, node_label, fillcolor=current_color)
            if parent_id:
                dot.edge(parent_id, key_id)
            build_flowchart(dot, value, key_id, level + 1)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                build_flowchart(dot, item, parent_id, level + 1)
            else:
                item_label = str(item).strip()
                item_id = hash_id(f"{parent_id}_{item_label}")
                add_node(dot, item_id, f"• {item_label}", shape='box', style='filled', fillcolor='white')
                if parent_id:
                    dot.edge(parent_id, item_id)
    else:
        if parent_id and str(data).strip():
            val_label = str(data).strip()
            val_id = hash_id(f"{parent_id}_{val_label}")
            add_node(dot, val_id, val_label, shape='ellipse', fillcolor='white')
            dot.edge(parent_id, val_id)

# --- CHANGE 2: Configure API key from environment variables ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")
genai.configure(api_key=api_key)

# --- CHANGE 3: Add safety settings and generation config for reliability ---
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
generation_config = {"response_mime_type": "application/json"}

gemini_model = genai.GenerativeModel(
    "models/gemini-1.5-pro",
    safety_settings=safety_settings,
    generation_config=generation_config
)
gemini_model_text = genai.GenerativeModel("models/gemini-1.5-pro", safety_settings=safety_settings) # For non-JSON prompts

# --- CHANGE 4: Improve error logging in the API call function ---
def call_gemini_api(prompt, model=gemini_model_text):
    try:
        response = model.generate_content(prompt)
        # Clean markdown ```json ``` fences if the model adds them
        cleaned_text = re.sub(r'^```json\s*|```\s*$', '', response.text.strip(), flags=re.MULTILINE)
        return cleaned_text
    except Exception as e:
        print(f"Gemini API Error: {e}") # This will print the actual error to your console
        return None

def generate_recommendation_from_input(user_interest, user_qualification):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, "final_course_data_for_bert.csv")
        df = pd.read_csv(csv_path)
        df['course_summary'] = "Course: " + df['Course'] + " | Field: " + df['Field']

        model = SentenceTransformer('all-MiniLM-L6-v2')

        df_filtered = df[df['Minimum_Qualification'].str.lower() == user_qualification.lower()]
        if df_filtered.empty:
            return {"error": "No courses found for your qualification."}

        course_texts = df_filtered['course_summary'].tolist()
        course_embeddings = model.encode(course_texts, show_progress_bar=True)
        user_embedding = model.encode([user_interest])

        similarities = cosine_similarity(user_embedding, course_embeddings)[0]
        best_index = similarities.argmax()
        best_row = df_filtered.iloc[best_index]
        best_course = best_row['Course']

        course_prefix = best_course.split()[0].upper()
        prompt_related = (
            f"Suggest 4 other full {course_prefix} engineering degree courses that are closely related to '{best_course}'. "
            "Return only a raw JSON array of strings, like [\"Course A\", \"Course B\"]."
        )

        # Use the model configured for JSON output
        related_courses_response = call_gemini_api(prompt_related, model=gemini_model)
        if not related_courses_response:
            return {"error": "Failed to get related course suggestions."}

        try:
            related_courses = json.loads(related_courses_response)
            if not isinstance(related_courses, list):
                raise ValueError("Gemini did not return a list.")
        except Exception as e:
            return {"error": f"Gemini response parsing error: {e}. Response was: {related_courses_response}"}

        all_courses = [best_course] + related_courses
        selected_course = all_courses[0]

        prompt_roadmap = f"""
        You are an expert mentor helping a student succeed in the course '{selected_course}'.
        Create a deeply structured 4-year roadmap from scratch to expert level.
        Format everything as a single JSON object. The root key must be 'roadmap'.
        Inside 'roadmap', include a 'title' with the course name.
        Do NOT add any text or explanations outside of the main JSON object.
        """

        # Use the model configured for JSON output
        roadmap_json_str = call_gemini_api(prompt_roadmap, model=gemini_model)
        if not roadmap_json_str:
            return {"error": "Failed to generate roadmap."}

        try:
            json_data = json.loads(roadmap_json_str)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON for roadmap from Gemini: {e}. Response was: {roadmap_json_str}"}

        title = json_data.get("roadmap", {}).get("title", "Career_Roadmap")
        dot = Digraph(comment=title,
                      graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'size': '20,30', 'ratio': 'compress'},
                      node_attr={'fontname': 'Helvetica', 'fontsize': '12', 'width': '1.5', 'height': '0.8'},
                      edge_attr={'fontsize': '10'})

        title_id = hash_id('title')
        dot.node(title_id, title, shape='doubleoctagon', style='filled', fillcolor='dodgerblue',
                 fontcolor='white', fontsize='18', width='2.0', height='1.0')

        build_flowchart(dot, json_data.get("roadmap"), title_id)
        
        # Ensure the 'outputs' directory exists
        output_dir = os.path.join(BASE_DIR, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, sanitize_filename(title))
        
        dot.render(output_path, format="pdf", cleanup=True)

        return {
            "recommended_course": best_course,
            "related_courses": all_courses,
            "roadmap": json_data.get("roadmap"),
            "pdf_path": f"{output_path}.pdf"
        }

    except Exception as e:
        # For debugging, it's helpful to see the full traceback in the console
        import traceback
        traceback.print_exc()
        return {"error": str(e)}