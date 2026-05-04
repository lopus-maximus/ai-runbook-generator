import cv2
import os
import time
import json
import subprocess

from PyPDF2 import PdfReader
from .models import Frame

from PIL import Image
import imagehash
import pytesseract

from google import genai
from google.genai import types

from faster_whisper import WhisperModel


client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# -------------------------------
# WHISPER MODEL (LOAD ONCE)
# -------------------------------

whisper_model = WhisperModel("base", device="cpu", compute_type="int8")


# -------------------------------
# AUDIO EXTRACTION FROM VIDEO
# -------------------------------

def extract_audio(video_path):
    audio_path = os.path.splitext(video_path)[0] + "_audio.wav"
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path,
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


# -------------------------------
# TRANSCRIPTION VIA WHISPER
# -------------------------------

def transcribe(audio_path):
    """
    Transcribe audio and auto-translate to English if the detected language
    is not English.

    Returns:
      {
        "transcript": str,
        "detected_language": str,
        "was_translated": bool,
      }
    """
    _, info = whisper_model.transcribe(audio_path, beam_size=1)
    detected_language = info.language

    if detected_language == "en":
        segments, _ = whisper_model.transcribe(audio_path)
        transcript    = " ".join(seg.text for seg in segments)
        was_translated = False
    else:
        segments, _ = whisper_model.transcribe(audio_path, task="translate")
        transcript    = " ".join(seg.text for seg in segments)
        was_translated = True

    return {
        "transcript":        transcript,
        "detected_language": detected_language,
        "was_translated":    was_translated,
    }


# -------------------------------
# FRAME EXTRACTION
# -------------------------------

def extract_frames(video_path, meeting):
    start_time    = time.time()
    meeting.status = "processing"
    meeting.save()

    cap   = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    os.makedirs("media/frames", exist_ok=True)

    while True:
        success, frame = cap.read()
        if not success:
            break

        if count % 30 == 0:
            frame_path = f"frames/frame_{meeting.id}_{count}.jpg"
            cv2.imwrite(f"media/{frame_path}", frame)
            Frame.objects.create(meeting=meeting, image=frame_path)
            saved += 1

        count += 1

    cap.release()

    meeting.status          = "processed"
    meeting.frame_count     = saved
    meeting.processing_time = round(time.time() - start_time, 2)
    meeting.save()


# -------------------------------
# PDF TEXT EXTRACTION
# -------------------------------

def extract_pdf_text(pdf_path):
    if not pdf_path:
        return ""
    reader = PdfReader(pdf_path)
    text   = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# -------------------------------
# DUPLICATE FRAME REMOVAL
# -------------------------------

def remove_duplicate_frames(frames_dir="media/frames", threshold=4):
    frame_files = sorted(os.listdir(frames_dir))
    last_hash   = None
    removed     = 0

    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        try:
            img          = Image.open(frame_path)
            current_hash = imagehash.phash(img)

            if last_hash is not None and abs(current_hash - last_hash) < threshold:
                os.remove(frame_path)
                Frame.objects.filter(image=f"frames/{frame_file}").delete()
                removed += 1
                continue

            last_hash = current_hash
        except Exception as e:
            print(f"Skipping {frame_file}: {e}")

    print(f"Duplicate frames removed: {removed}")


# -------------------------------
# SMART FRAME SELECTION
# -------------------------------

def smart_select_frames(frames_dir="media/frames", step=10, top_k=5):
    frame_files = sorted(os.listdir(frames_dir))
    sampled     = frame_files[::step]
    scored      = []

    for f in sampled:
        path = os.path.join(frames_dir, f)
        try:
            text  = pytesseract.image_to_string(Image.open(path))
            words = text.split()
            if len(words) > 20 and len(set(words)) > 10:
                scored.append((f, len(words)))
        except Exception:
            continue

    scored.sort(key=lambda x: x[1], reverse=True)
    return [f"frames/{x[0]}" for x in scored[:top_k]]


# -------------------------------
# GEMINI IMAGE LOADING
# -------------------------------

def load_images_for_gemini(frame_paths):
    parts = []
    for frame in frame_paths:
        full_path = os.path.join("media", frame)
        with open(full_path, "rb") as f:
            img_bytes = f.read()
        parts.append(
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
        )
    return parts


# -------------------------------
# TEMPLATE NORMALISATION
# -------------------------------

def normalize_template_with_gemini(template_text):
    prompt = (
        "Convert this template into STRICT JSON format.\n\n"
        "Return ONLY valid JSON:\n\n"
        '{\n  "sections": [\n    {\n      "heading": ""\n    }\n  ]\n}\n\n'
        "Rules:\n"
        "- Preserve original headings\n"
        "- Do not invent new sections\n"
        "- No explanations\n"
        "- No markdown\n\n"
        "Template:\n" + template_text
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )

    raw   = (response.text or "").strip()
    start = raw.find("{")
    end   = raw.rfind("}")

    if start == -1 or end == -1:
        raise Exception("Template normalization failed")

    return json.loads(raw[start : end + 1])


# -------------------------------
# AI RUNBOOK GENERATION
# -------------------------------

def generate_runbook_ai(text, selected_frames, template_schema):
    image_parts   = load_images_for_gemini(selected_frames)
    frame_list    = "\n".join(selected_frames)
    sections_json = json.dumps(template_schema.get("sections", []), indent=2)

    prompt = (
        "Generate a detailed meeting runbook.\n\n"
        "Available images (use ONLY these filenames):\n\n"
        + frame_list
        + "\n\nReturn STRICT JSON in this structure:\n\n"
        "{\n"
        '  "title": "",\n'
        '  "sections": [\n'
        '    {\n'
        '      "heading": "",\n'
        '      "paragraphs": [""],\n'
        '      "supporting_images": [\n'
        '        { "image": "", "caption": "" }\n'
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "key_decisions": [],\n'
        '  "action_items": [\n'
        '    { "task": "", "owner": "", "deadline": "" }\n'
        "  ],\n"
        '  "risks_or_blockers": [],\n'
        '  "next_steps": []\n'
        "}\n\n"
        "Use ONLY these section headings:\n\n"
        + sections_json
        + "\n\nRules:\n"
        "- Focus on WRITTEN CONTENT first.\n"
        "- Each section must contain meaningful paragraphs.\n"
        "- Images are OPTIONAL — use only where genuinely relevant.\n"
        "- Do NOT rename or invent image filenames.\n"
        "- No markdown. No explanations outside JSON.\n\n"
        "Meeting content:\n" + text
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=prompt), *image_parts],
        )
    ]

    for attempt in range(1):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
            )
            raw = (response.text or "").strip()
            if not raw:
                raise Exception("Empty Gemini response")

            start = raw.find("{")
            end   = raw.rfind("}")
            if start == -1 or end == -1:
                raise Exception("No JSON object found")

            return json.loads(raw[start : end + 1])

        except Exception as e:
            print(f"Gemini error (attempt {attempt + 1}): {e}")
            time.sleep(2)

    raise Exception("Gemini failed after retries")


# -------------------------------
# AI RUNBOOK EDITING
# -------------------------------

def edit_runbook_ai(instruction: str, current_runbook: dict) -> dict:
    """
    Takes a plain-English instruction and the current runbook JSON.
    Sends both to Gemini and returns an updated runbook dict.
    """
    current_json = json.dumps(current_runbook, indent=2)

    prompt = (
        "You are an expert document editor. You will be given an existing meeting runbook "
        "in JSON format and a plain-English instruction describing changes to make.\n\n"
        "Apply the requested changes and return the COMPLETE updated runbook as STRICT JSON.\n\n"
        "Rules:\n"
        "- Preserve the exact same JSON structure and all existing fields.\n"
        "- Apply ONLY the changes described in the instruction — do not alter unrelated content.\n"
        "- If the instruction asks to add a section, add it to the sections array with heading and paragraphs.\n"
        "- If the instruction asks to remove content, remove it cleanly.\n"
        "- Keep all existing supporting_images references intact unless told to remove them.\n"
        "- Return ONLY valid JSON. No markdown fences, no explanations outside the JSON.\n\n"
        "=== CURRENT RUNBOOK JSON ===\n"
        + current_json
        + "\n\n=== INSTRUCTION ===\n"
        + instruction
        + "\n\n=== UPDATED RUNBOOK JSON ==="
    )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            raw = (response.text or "").strip()
            if not raw:
                raise Exception("Empty Gemini response")

            start = raw.find("{")
            end   = raw.rfind("}")
            if start == -1 or end == -1:
                raise Exception("No JSON object found in response")

            return json.loads(raw[start : end + 1])

        except Exception as e:
            print(f"edit_runbook_ai error (attempt {attempt + 1}): {e}")
            time.sleep(2)

    raise Exception("Gemini failed to apply edits after retries")


# -------------------------------
# AI Q&A
# -------------------------------

def answer_question_ai(question: str, context: str) -> str:
    """
    Answer a question grounded strictly in the meeting context
    (transcript + PDF text).  Returns a plain-text answer string.
    """
    prompt = (
        "You are an assistant that answers questions about a meeting.\n"
        "Answer ONLY using the information in the meeting context below.\n"
        "If the answer is not in the context, say so clearly.\n"
        "Be concise and direct. No markdown.\n\n"
        "=== MEETING CONTEXT ===\n"
        + context
        + "\n\n=== QUESTION ===\n"
        + question
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )

    return (response.text or "I couldn't find an answer in the meeting content.").strip()