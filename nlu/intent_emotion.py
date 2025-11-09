from transformers import pipeline

# Pretrained HuggingFace models (no fine-tune yet)
intent_pipe = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

def analyze_text(user_input):
    intent = intent_pipe(user_input)[0]
    emotion = emotion_pipe(user_input)[0]
    return {
        "intent": intent["label"],
        "emotion": emotion["label"],
        "confidence": (intent["score"] + emotion["score"]) / 2
    }
