import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch, whisper, base64, json, random
from flask import Flask, send_from_directory, Response, request, jsonify
from diffusers import AutoPipelineForText2Image
from io import BytesIO
from backend import generate_frames

app = Flask(__name__, static_folder="studioassistant/build", static_url_path="/")
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
whisper = whisper.load_model("tiny")

with open("questions.json", "r", encoding="utf-8") as file:
    QUESTIONS = json.load(file)

@app.route("/api/question", methods=["GET"])
def get_random_question():
    """Returns a random question from the dataset"""
    question_data = random.choice(QUESTIONS)
    return jsonify({
        "question": question_data["question"]["stem"],
        "choices": question_data["question"]["choices"],
        "answer": question_data["answerKey"]
    })


@app.route("/")
def index():
    return send_from_directory("studioassistant/build", "index.html")

# Serve other static files (CSS, JS, images)
@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("studioassistant/build", path)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/audio-upload", methods=["POST"])
def audio_upload():
    audio_file = request.files["file"]
    audio_file.save("audio.wav")
    
    # Speech-to-text
    transcription = whisper.transcribe("audio.wav")
    text = transcription["text"]
    
    return jsonify({"transcribed_text": text})

@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.json
    prompt = data["prompt"]
    
    # Generate image
    neg_prompt = "ugly, blurry, poor quality, deformed structure, very bad lighting, bad colouring, noise"
    image = pipe(prompt=prompt, negative_prompt=neg_prompt, num_inference_steps=5, height=256, width=256).images[0]
    
    # Convert image to base64 for frontend display
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return jsonify({"image": img_str})

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000, threaded=True, use_reloader=False)
