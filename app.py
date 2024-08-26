import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import string
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import torch
from gtts import gTTS
import language_tool_python

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['AUDIO_FOLDER'] = 'audio'

# Create the audio folder if it doesn't exist
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

def correct_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    corrected_text = tool.correct(text)
    return corrected_text

def image2text(image_path, conditional_keyword):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    raw_image = Image.open(image_path).convert('RGB')

    # Conditional image captioning
    text = conditional_keyword if conditional_keyword else "The"
    inputs = processor(raw_image, text, return_tensors="pt")
    min_tokens = 30
    out = model.generate(**inputs, min_length=min_tokens, max_length=min_tokens+5)
    conditional_description = processor.decode(out[0], skip_special_tokens=True)

    # Unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs, min_length=min_tokens, max_length=min_tokens+5)
    unconditional_description = processor.decode(out[0], skip_special_tokens=True)

    # Apply grammar correction
    corrected_conditional_description = correct_grammar(conditional_description.capitalize())
    corrected_unconditional_description = correct_grammar(unconditional_description.capitalize())

    return corrected_conditional_description, corrected_unconditional_description

def text2speech(text, filename):
    audio_path = os.path.join(app.config['AUDIO_FOLDER'], filename)
    tts = gTTS(text)
    tts.save(audio_path)
    return audio_path

def sentiment(text):
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", device=device)
    result = pipe(text)
    emotions_list = [{"label": entry['label'], "score": round(entry['score'] * 100, 3)} for entry in result]
    return emotions_list

def automatic_speech_recognition_function(audio_path):
    device = 0 if torch.cuda.is_available() else -1
    asr_pipeline = pipeline("automatic-speech-recognition", model="distil-whisper/distil-large-v3", device=device)
    result = asr_pipeline(audio_path)
    transcribed_text = result['text']
    return transcribed_text

##################################################

@app.route('/', methods=['GET', 'POST'])
def start_point():
    return render_template('bootstrap_website.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files or 'conditional_keyword' not in request.form:
            return jsonify({"conditional_audio_path": None, "unconditional_audio_path": None,
                            "conditional_description": "No image or keyword uploaded", "unconditional_description": "No image uploaded"})

        image = request.files['image']
        conditional_keyword = request.form['conditional_keyword']

        if image.filename == '':
            return jsonify({"conditional_audio_path": None, "unconditional_audio_path": None,
                            "conditional_description": "No selected image", "unconditional_description": "No selected image"})

        if image:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
            image.save(image_path)
            conditional_description, unconditional_description = image2text(image_path, conditional_keyword)

            conditional_audio_path = text2speech(conditional_description, 'conditional_audio.mp3')
            unconditional_audio_path = text2speech(unconditional_description, 'unconditional_audio.mp3')

            response_data = {
                "conditional_description": conditional_description,
                "unconditional_description": unconditional_description,
                "conditional_audio_path": conditional_audio_path,
                "unconditional_audio_path": unconditional_audio_path,
            }

            return jsonify(response_data)

    return render_template('index.html')

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment_analysis():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if not user_input:
            return jsonify({"error": "Please provide 'user_input' parameter in the form data."}), 400

        sentiment_result = sentiment(user_input)
        formatted_sentiment_result = [f"{entry['label'].capitalize()} : {entry['score']:.3f}" for entry in sentiment_result]
        result_string = ", ".join(formatted_sentiment_result)
    
        return jsonify({"sentiment": result_string})
    return render_template('sentiment.html')

@app.route('/automatic_speech_recognition', methods=['GET', 'POST'])
def automatic_speech_recognition():
    if request.method == 'POST':
        if 'audio' not in request.files:
            return jsonify({"transcribed_text": "No audio file uploaded"})

        audio_file = request.files['audio']

        if audio_file.filename == '':
            return jsonify({"transcribed_text": "No selected audio file"})

        if audio_file:
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_audio.mp3')
            audio_file.save(audio_path)

            transcribed_text = automatic_speech_recognition_function(audio_path)

            transcribed_text_cleaned = ''.join(char for char in transcribed_text if char not in string.punctuation)

            response_data = {
                "transcribed_text": transcribed_text_cleaned,
                "original_transcription": transcribed_text,
            }

            return jsonify(response_data)

        return jsonify({"transcribed_text": "Unexpected error during audio processing"})

    return render_template('automatic_speech_recognition.html')

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['AUDIO_FOLDER'], filename)

@app.route('/text_to_audio', methods=['GET', 'POST'])
def text_to_audio():
    if request.method == 'POST':
        text = request.form['text']
        tts = gTTS(text)
        output_path = os.path.join(app.config['AUDIO_FOLDER'], 'output.mp3')
        tts.save(output_path)
        return send_file(output_path, as_attachment=True)
    return render_template('text_to_audio.html')

if __name__ == '__main__':
    app.run(debug=True)
