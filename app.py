import os
import tempfile
import json

try:
    import ollama
except ImportError:
    print("download ollama")
    exit()

print("mechanical design")

system_instructions = (
    "You are an autonomous mechanical engineering AI. Your job is to invent a useful, real-world mechanical part or component. "
    "You must output ONLY a raw JSON object with two keys:\n"
    "1. 'part_name': A short string naming what you invented.\n"
    "2. 'cadquery_code': A string containing valid, executable CadQuery python code that creates the 3D model and saves it.\n\n"
    "CRITICAL RULES FOR THE CADQUERY CODE:\n"
    "- Use simple primitives like box(), circle(), extrude(), hole(), or loft().\n"
    "- Always name the final 3D object variable 'final_model'.\n"
    "- Do not include any file export lines (like cq.exporters.export) inside the code string; the host script handles saving.\n"
    "- Keep the code reliable, completely avoiding complex face selections or hard edge fillets that might break.\n\n"
    "Example Output Format:\n"
    '{"part_name": "Spacer Block", "cadquery_code": "import cadquery as cq\\nfinal_model = cq.Workplane(\\"XY\\").box(50, 50, 10).faces(\\">Z\\").workplane().hole(12)"}'
)

try:
    response = ollama.generate(
        model="llama3",
        system=system_instructions,
        prompt="Invent a random mechanical component (e.g., a custom bracket, a pulley, a hinge plate, a specialized connector) and write its CAD code.",
        options={"temperature": 0.9}
    )
    
    clean_text = response['response'].strip()
    if "```json" in clean_text:
        clean_text = clean_text.split("```json")[1].split("```")[0].strip()
    elif "```" in clean_text:
        clean_text = clean_text.split("```")[1].strip()
        
    ai_data = json.loads(clean_text)
    part_name = ai_data.get("part_name", "AI_Generated_Part")
    cad_script = ai_data.get("cadquery_code", "")
    
except Exception as e:
    print(f"Ollama generation stalled. Make sure 'ollama run llama3' works in terminal. Error: {e}")
    exit()

user_home = os.environ["USERPROFILE"].replace("\\", "/")
output_dir = f"{user_home}/Downloads"
step_path = f"{output_dir}/ai_generated_part.step"
stl_path = f"{output_dir}/ai_generated_part.stl"

macro_code = f"""{cad_script}

try:
    cq.exporters.export(final_model, "{step_path}")
    cq.exporters.export(final_model, "{stl_path}")
except NameError:
    print("Error: The AI script did not define 'final_model' properly.")
"""

temp_dir = tempfile.gettempdir().replace("\\", "/")
macro_path = f"{temp_dir}/temp_ai_factory.py"

with open(macro_path, "w") as f:
    f.write(macro_code)

try:
    os.system(f"python \"{macro_path}\"")
    print(f"\n CAD Model!")
    print(f"Engine Invented: {part_name.upper()}")
    print(f"Saved your new model to: Downloads/ai_generated_part.stl")
except Exception as e:
    print(f"CAD compilation error: {e}")
