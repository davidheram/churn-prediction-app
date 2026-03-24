from openai import OpenAI
import os 
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2]/".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("No se encontró OPENAI_API_KEY")


client = OpenAI(api_key=api_key)

def generate_explanation(input_data, proba): 
    prompt   =f"""
    Eres un analista experto en churn. 

    Datos del cliente:

    {input_data.to_dict(orient="records")[0]}

    Probabilidad de churn: {proba:.2f}

    Explica:
    1. Por qué el cliente podría abandonar 
    2. Que acciones tomar para retenerlo 

    Sé claro, corto y accionable.
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role":"system", "content": "Eres un experto en retencion de clientes."},
            {"role":"user","content":prompt},

        ],
    )
    return response.choices[0].message.content

