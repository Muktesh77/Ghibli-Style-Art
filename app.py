import streamlit as st
import torch
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
import os
import io
import datetime
import ollama_addition # New import for Ollama

# --- 1. Streamlit Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Ghibli Style AI Generator",
    page_icon="✨",
    layout="centered"
)


# --- 2. Configuration Centralization ---
CONFIG = {
    "ghibli_model_id": "nitrosocke/Ghibli-Diffusion",
    "output_dir": "generated_images_frontend",
    "default_prompt": "A serene landscape with a towering ancient tree, surrounded by whispering spirits, Studio Ghibli film, by Hayao Miyazaki, anime studio art, highly detailed, beautiful lighting, cinematic",
    "default_negative_prompt": "low quality, bad anatomy, deformed, ugly, disfigured, blurry, realistic, photo, CGI, 3D, text, signature",
    "default_inference_steps": 30,
    "min_inference_steps": 10,
    "max_inference_steps": 50,
    "default_guidance_scale": 7.5,
    "min_guidance_scale": 0.0,
    "max_guidance_scale": 15.0,
    "step_guidance_scale": 0.5,
    "default_img2img_strength": 0.55,
    "min_img2img_strength": 0.0,
    "max_img2img_strength": 1.0,
    "step_img2img_strength": 0.05,
    "image_resize_dims": (512, 512),
    "random_seed": 42,
    "ollama_model_name": "mistral" # Default Ollama model to use for prompt generation
}

# Ensure output directory exists
os.makedirs(CONFIG["output_dir"], exist_ok=True)


# --- 3. Model Loading (Cached) ---
@st.cache_resource
def load_models():
    """
    Loads the Diffusion pipelines for text-to-image and image-to-image generation.
    Models are cached to avoid reloading on every Streamlit interaction.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    if device == "cuda":
        st.write("Using GPU for inference locally (optimized for speed).")
    else:
        st.warning("Using CPU for inference locally. This will be significantly slower. Consider a GPU for better performance.")

    try:
        pipe_txt2img = DiffusionPipeline.from_pretrained(CONFIG["ghibli_model_id"], torch_dtype=dtype).to(device)
        pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(CONFIG["ghibli_model_id"], torch_dtype=dtype).to(device)
        st.success("AI models loaded successfully!")
        return pipe_txt2img, pipe_img2img, device
    except Exception as e:
        st.error(f"Failed to load AI models! Please ensure you have enough RAM/VRAM and correct dependencies.")
        st.exception(e)
        st.stop()


# Load models at the very beginning of the script execution (after config)
txt2img_pipeline, img2img_pipeline, device = load_models()


# --- 4. Modularized Image Generation Functions ---
def generate_image_from_text(pipeline, prompt, negative_prompt, num_inference_steps, guidance_scale, device, seed=None):
    """
    Generates an image from a text prompt using the provided pipeline.
    """
    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

    with st.spinner("Generating image from text... This may take a while."):
        try:
            image = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
            st.success("Image generated successfully!")
            return image
        except Exception as e:
            st.error(f"Error during text-to-image generation: {e}")
            st.exception(e)
            return None

def generate_image_from_image(pipeline, init_image, prompt, negative_prompt, num_inference_steps, guidance_scale, strength, device, seed=None):
    """
    Converts an input image to Ghibli style using the provided pipeline.
    """
    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

    with st.spinner("Converting image to Ghibli style... This may take a while."):
        try:
            image = pipeline(
                prompt=prompt,
                image=init_image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator
            ).images[0]
            st.success("Image converted successfully!")
            return image
        except Exception as e:
            st.error(f"Error during image-to-image conversion: {e}")
            st.exception(e)
            return None


# --- 5. Ollama Integration Function ---
@st.cache_data(show_spinner=False) # Cache Ollama responses to avoid repeated calls for same input
def generate_prompt_with_ollama(user_keywords, llm_model_name):
    """
    Interacts with Ollama to generate a more detailed image prompt.
    """
    if not user_keywords.strip():
        return "" # Return empty if no keywords provided

    ollama_prompt = f"""
    You are an AI assistant that generates detailed, creative prompts for a Studio Ghibli style image generator.
    The user will provide a few keywords or a short idea. Expand on these keywords to create a vivid, descriptive prompt that emphasizes Studio Ghibli aesthetic elements.
    Ensure the prompt is concise enough for image generation but rich in detail. Do NOT include phrases like "Studio Ghibli style" or "by Hayao Miyazaki" as those will be added automatically by the main app.
    Focus on scene, characters, lighting, colors, and atmosphere.

    Keywords: {user_keywords}

    Detailed Ghibli-esque Prompt:
    """

    try:
        response = ollama_addition.chat(model=llm_model_name, messages=[{'role': 'user', 'content': ollama_prompt}])
        return response['message']['content'].strip()
    except ollama_addition.ResponseError as e:
        st.error(f"Ollama Error: Could not connect to Ollama server or model '{llm_model_name}' not found. Please ensure Ollama is running and the model is downloaded. Details: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred with Ollama: {e}")
        return None

# --- 6. Streamlit UI Structure ---
st.title("✨ Ghibli Style AI Generator ✨")
st.markdown("Convert your ideas or photos into beautiful Studio Ghibli art!")

# Prompt Generation Assistant (New Section)
with st.expander("✨ AI Prompt Assistant (powered by Ollama)", expanded=False):
    st.write("Let an AI help you craft a creative prompt for your Ghibli image!")
    ollama_keywords = st.text_input(
        "Enter keywords or a brief idea for your Ghibli image:",
        placeholder="e.g., magical forest, lonely robot, soaring dragon"
    )
    # Dynamically list available Ollama models
    available_ollama_models = []
    try:
        # Fetching available models - this requires Ollama server to be running
        models_list = ollama_addition.list()['models']
        # --- CORRECTED LINE BELOW: Changed 'm['model']' to 'm['name']' for filtering ---
        available_ollama_models = [m for m in models_list if m['name'].startswith(('mistral', 'llama3', 'gemma'))] # Filter for relevant models
    except Exception as e:
        st.warning(f"Could not connect to Ollama server or list models. Please ensure Ollama is running: {e}")
        st.info("You might need to download a model (e.g., `ollama pull mistral`) in your terminal first.")

    # Default to mistral if available, otherwise pick the first or 'None'
    default_ollama_index = 0
    if available_ollama_models:
        try:
            # Find index of the default model from CONFIG
            default_ollama_index = next((i for i, model in enumerate(available_ollama_models) if model['name'] == CONFIG["ollama_model_name"]), 0)
        except StopIteration:
            # If default model name not in list, fallback to first model
            default_ollama_index = 0

    ollama_model_selector = st.selectbox(
        "Select Ollama Model:",
        options=available_ollama_models,
        format_func=lambda x: x['name'], # This is where the KeyError was originating due to malformed 'model' object
        index=default_ollama_index,
        key="ollama_model_selector" # Unique key for this widget
    )

    if st.button("Generate Detailed Prompt with AI", use_container_width=True, key="generate_ollama_prompt_btn"): # Unique key
        if ollama_keywords and ollama_model_selector: # Ensure a model is selected
            with st.spinner(f"Asking {ollama_model_selector['name']} to refine your prompt..."):
                generated_llm_prompt = generate_prompt_with_ollama(ollama_keywords, ollama_model_selector['name'])
                if generated_llm_prompt:
                    st.session_state['main_prompt_text'] = generated_llm_prompt
                    st.success("Prompt generated! Check the main prompt box below.")
                else:
                    st.error("Could not generate prompt. Check Ollama server and model, or try different keywords.")
        else:
            st.warning("Please enter some keywords and ensure an Ollama model is selected.")

# This trick ensures the main prompt area updates after LLM generation
if 'main_prompt_text' not in st.session_state:
    st.session_state['main_prompt_text'] = CONFIG["default_prompt"]

# Mode Selection
mode = st.radio(
    "Choose Generation Mode:",
    ("Text-to-Image", "Image-to-Image Style Transfer"),
    key="generation_mode_selector" # Unique key
)

# Common Parameters in a container for better grouping
with st.container(border=True):
    st.markdown("### Common Generation Settings")

    prompt_input = st.text_area(
        "Enter your prompt:",
        value=st.session_state['main_prompt_text'],
        key="main_prompt_area" # Unique key for this widget
    )

    negative_prompt_input = st.text_area(
        "Enter negative prompt (optional):",
        CONFIG["default_negative_prompt"],
        key="negative_prompt_area" # Unique key
    )

    col1, col2 = st.columns(2)
    with col1:
        num_inference_steps = st.slider(
            "Number of Inference Steps:",
            CONFIG["min_inference_steps"],
            CONFIG["max_inference_steps"],
            CONFIG["default_inference_steps"],
            key="inference_steps_slider" # Unique key
        )
    with col2:
        guidance_scale = st.slider(
            "Guidance Scale (how much prompt adherence):",
            CONFIG["min_guidance_scale"],
            CONFIG["max_guidance_scale"],
            CONFIG["default_guidance_scale"],
            CONFIG["step_guidance_scale"],
            key="guidance_scale_slider" # Unique key
        )

generated_image = None

if mode == "Text-to-Image":
    st.subheader("Text-to-Image Generation")
    if st.button("Generate Ghibli Image from Text", use_container_width=True, key="generate_txt2img_btn"): # Unique key
        generated_image = generate_image_from_text(
            txt2img_pipeline,
            prompt_input,
            negative_prompt_input,
            num_inference_steps,
            guidance_scale,
            device,
            seed=CONFIG["random_seed"]
        )

elif mode == "Image-to-Image Style Transfer":
    st.subheader("Image-to-Image Style Transfer")

    uploaded_file = st.file_uploader("Upload an image to convert to Ghibli style:", type=["png", "jpg", "jpeg"], key="image_uploader") # Unique key

    strength_input = st.slider(
        "Strength (how much to change the original image, 0.0-1.0):",
        CONFIG["min_img2img_strength"],
        CONFIG["max_img2img_strength"],
        CONFIG["default_img2img_strength"],
        CONFIG["step_img2img_strength"],
        key="strength_slider" # Unique key
    )

    init_image = None
    if uploaded_file is not None:
        try:
            init_image = Image.open(uploaded_file).convert("RGB").resize(CONFIG["image_resize_dims"])
            st.image(init_image, caption="Original Image Preview", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
            st.exception(e)

    if init_image is not None and st.button("Convert Image to Ghibli Style", use_container_width=True, key="generate_img2img_btn"): # Unique key
        generated_image = generate_image_from_image(
            img2img_pipeline,
            init_image,
            prompt_input,
            negative_prompt_input,
            num_inference_steps,
            guidance_scale,
            strength_input,
            device,
            seed=CONFIG["random_seed"]
        )

# --- Display Generated Image & Save ---
if generated_image is not None:
    st.subheader("Generated Ghibli Image:")
    st.image(generated_image, caption="Ghibli Style Output", use_container_width=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"ghibli_output_{mode.replace(' ', '_').replace('-', '').lower()}_{timestamp}.png"
    output_path = os.path.join(CONFIG["output_dir"], output_filename)

    try:
        generated_image.save(output_path)
        st.success(f"Image saved successfully to: `{output_path}`")
    except Exception as e:
            st.error(f"Failed to save image: {e}")
            st.exception(e)

st.markdown("---")
st.markdown("Built with ❤️ and Streamlit")