import gradio as gr
from huggingface_hub import InferenceClient
import os
# --- Configuration ---
MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Initialize the client
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# --- System Prompt & Logic ---
SYSTEM_PROMPT = """Eres un tutor socrático experto y empático.
Tu objetivo es guiar al estudiante para que descubra las respuestas por sí mismo mediante el pensamiento crítico.
1. No des la respuesta directa.
2. Haz preguntas reflexivas y abiertas.
3. Mantén tus respuestas concisas (máximo 3 oraciones).
4. Usa un tono profesional pero alentador.
5. Responde siempre en español.
"""

def format_prompt(message, history):
    prompt = f"<s>[INST] {SYSTEM_PROMPT} [/INST]</s>"
    for msg in history:
        if msg['role'] == "user":
            prompt += f"<s>[INST] {msg['content']} [/INST]"
        elif msg['role'] == "assistant":
            prompt += f" {msg['content']} </s>"
    prompt += f"<s>[INST] {message} [/INST]"
    return prompt

def respond(message, history):
    if not HF_TOKEN:
        yield "⚠️ Error de configuración: HF_TOKEN no está definido. Por favor configura la variable de entorno."
        return

    formatted_prompt = format_prompt(message, history)

    generate_kwargs = dict(
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True,
    )

    try:
        stream = client.text_generation(formatted_prompt, stream=True, details=False, **generate_kwargs)
        partial_response = ""
        for response in stream:
            partial_response += response
            yield partial_response
    except Exception as e:
        yield f"❌ Error: {str(e)}. Intenta de nuevo más tarde."

# --- UI Design ---
custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.chat-message {
    border-radius: 12px !important;
    padding: 16px !important;
}
.user-message {
    background-color: #eff6ff !important; /* Blue-50 */
    border: 1px solid #dbeafe !important;
}
.bot-message {
    background-color: #f9fafb !important; /* Gray-50 */
    border: 1px solid #e5e7eb !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}
h1 {
    color: #1e40af; /* Blue-800 */
    text-align: center;
    margin-bottom: 0.5rem;
}
.description {
    text-align: center;
    color: #4b5563; /* Gray-600 */
    margin-bottom: 2rem;
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

with gr.Blocks(title="Tutor Socrático IA", theme=theme, css=custom_css) as demo:
    with gr.Column(elem_id="container"):
        gr.Markdown("# 🎓 Tutor Socrático IA", elem_classes="title")
        gr.Markdown(
            "Bienvenido. Soy tu tutor personal. Hazme preguntas y te ayudaré a explorar el conocimiento mediante el método socrático.",
            elem_classes="description"
        )

        chat_interface = gr.ChatInterface(
            fn=respond,
            type="messages",
            examples=[
                "¿Por qué el cielo es azul?",
                "Explícame la teoría de la relatividad",
                "¿Qué es la ética?",
                "Ayúdame a entender las derivadas"
            ]
        )

if __name__ == "__main__":
    demo.launch()
