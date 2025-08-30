# Mental Health Chatbot - MindBuddy (Enhanced)
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

print("Hello! I am MindBuddy, your AI mental health chatbot.")
print("You can talk to me about how you're feeling today.")
print("Type 'quit' to exit.\n")

# Training data
training_sentences = [
    "I am feeling sad", "I feel unhappy", "I am depressed",
    "I am anxious", "I feel nervous", "I am worried",
    "I am happy", "I feel joyful", "I am excited",
    "I am stressed", "I feel overwhelmed", "I have a lot of pressure",
    "I am angry", "I feel frustrated", "I am mad",
    "I am tired", "I feel exhausted", "I am sleepy"
]

training_labels = [
    "sad", "sad", "sad",
    "anxious", "anxious", "anxious",
    "happy", "happy", "happy",
    "stress", "stress", "stress",
    "angry", "angry", "angry",
    "tired", "tired", "tired"
]

# Train ML model
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_sentences)
model = MultinomialNB()
model.fit(X_train, training_labels)

# Responses dictionary with multiple options
responses = {
    "sad": [
        "I'm sorry you're feeling sad. Talking about it might help. What's troubling you?",
        "Feeling sad is tough. Do you want to share more about it?",
        "Sadness can weigh heavy. I'm here to listen if you want."
    ],
    "happy": [
        "That's wonderful! What made you feel happy today?",
        "Yay! Happiness is great. Want to tell me more?",
        "I'm glad you're feeling good! What brought this joy?"
    ],
    "stress": [
        "Stress can be tough. Try deep breathing or a short walk. Need some tips?",
        "Take a moment to relax. What’s causing the stress?",
        "Stress happens. Let's try to lighten your mind together."
    ],
    "anxious": [
        "Feeling anxious is normal. Can you tell me more about what worries you?",
        "It's okay to feel nervous sometimes. Want to share what's on your mind?",
        "Anxiety can be heavy. I'm here to listen."
    ],
    "tired": [
        "Rest is important. Make sure you're taking breaks and sleeping well.",
        "Sounds like you need some rest. Have you had enough sleep?",
        "Being tired can affect everything. Take a short break if you can."
    ],
    "angry": [
        "Anger is natural. Talking about it can help calm your mind.",
        "It’s okay to feel frustrated. Want to vent a little?",
        "Anger can be intense. Deep breaths help, do you want to try?"
    ],
    "sad_anxious": [
        "It seems you're feeling both sad and anxious. Take a deep breath. Want to talk about it?",
        "Sadness and anxiety together can feel overwhelming. I'm here for you."
    ],
    "happy_stress": [
        "Even when stressed, it's nice to have happy moments. Focus on what makes you happy!",
        "Happy moments can balance stress. What's one good thing today?"
    ],
    "default": [
        "I see. Can you tell me more about how you feel?",
        "I'm listening. Please share more about your feelings.",
        "Interesting. Can you explain a bit more?"
    ]
}

# Function to detect dual emotions
def detect_dual_emotion(user_input):
    emotions = []
    for key in ["sad", "happy", "stress", "anxious", "angry", "tired"]:
        if key in user_input:
            emotions.append(key)
    if len(emotions) == 2:
        combo = "_".join(sorted(emotions))
        if combo in responses:
            return random.choice(responses[combo])
    return None

# Chat loop
while True:
    user_input = input("You: ").lower()
    if user_input == "quit":
        print("MindBuddy: Take care! Remember, talking to someone you trust helps. Goodbye!")
        break

    # Check for dual emotion first
    dual_response = detect_dual_emotion(user_input)
    if dual_response:
        print("MindBuddy:", dual_response)
        continue

    # Predict emotion with ML
    X_input = vectorizer.transform([user_input])
    predicted_emotion = model.predict(X_input)[0]

    # Respond with random choice
    response = random.choice(responses.get(predicted_emotion, responses["default"]))
    print("MindBuddy:", response)
