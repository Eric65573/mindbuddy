# Super-Enhanced MindBuddy - AI Mental Health Chatbot
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

print("Hello! I am MindBuddy, your AI mental health chatbot.")
print("You can talk to me about how you're feeling today.")
print("Type 'quit' to exit.\n")

# Expanded training sentences
training_sentences = [
    # Emotional phrases
    "I am feeling sad", "I feel unhappy", "I am depressed", "so sad", "very sad", "extremely sad",
    "I am anxious", "I feel nervous", "I am worried", "so anxious", "very anxious", "totally anxious",
    "I am happy", "I feel joyful", "I am excited", "super happy", "very happy", "extremely happy",
    "I am stressed", "I feel overwhelmed", "I have a lot of pressure", "totally stressed", "very stressed",
    "I am angry", "I feel frustrated", "I am mad", "so angry", "very angry",
    "I am tired", "I feel exhausted", "I am sleepy", "so tired", "completely exhausted",
    # Greetings
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening", "yo", "hiya", "greetings", "howdy",
    # Small talk / chit chat
    "how are you", "how's it going", "how are you doing", "what's up", "how have you been",
    "what are you doing", "what's new", "how's your day", "how's life",
    # Gratitude
    "thank you", "thanks", "thanks a lot", "thank you so much", "thx", "much appreciated", "thanks buddy",
    # Affirmations
    "yes", "yeah", "yep", "sure", "absolutely", "definitely", "of course", "roger", "correct", "true",
    # Negations
    "no", "nope", "nah", "not really", "never", "incorrect", "false",
    # Help/Advice
    "help me", "what should I do", "any advice", "can you help", "what can I do", "give me tips",
    # Fun / Casual
    "lol", "haha", "hahaha", "omg", "wow", "yikes", "oops"
]

# Corresponding labels
training_labels = [
    # Emotional labels
    "sad","sad","sad","sad","sad","sad",
    "anxious","anxious","anxious","anxious","anxious","anxious",
    "happy","happy","happy","happy","happy","happy",
    "stress","stress","stress","stress","stress",
    "angry","angry","angry","angry","angry",
    "tired","tired","tired","tired","tired",
    # Greetings
    "greeting","greeting","greeting","greeting","greeting","greeting","greeting","greeting","greeting","greeting",
    # Small talk
    "smalltalk","smalltalk","smalltalk","smalltalk","smalltalk","smalltalk","smalltalk","smalltalk","smalltalk",
    # Gratitude
    "gratitude","gratitude","gratitude","gratitude","gratitude","gratitude","gratitude",
    # Affirmations
    "affirmation","affirmation","affirmation","affirmation","affirmation","affirmation","affirmation","affirmation","affirmation","affirmation",
    # Negations
    "negation","negation","negation","negation","negation","negation","negation",
    # Help / Advice
    "help","help","help","help","help","help",
    # Fun / Casual
    "fun","fun","fun","fun","fun","fun","fun"
]

# Train ML model
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_sentences)
model = MultinomialNB()
model.fit(X_train, training_labels)

# Responses dictionary
responses = {
    "sad":[
        "I'm sorry you're feeling sad. Do you want to talk about it?",
        "Feeling sad can be heavy. I'm here to listen.",
        "It's okay to feel down sometimes. What's bothering you?",
        "Sadness is normal. Can you tell me more?",
        "I understand it can be tough. Want to share your thoughts?"
    ],
    "happy":[
        "Yay! That's great to hear! What made you happy?",
        "Happiness is wonderful! Tell me more!",
        "I'm glad you're feeling good! What brought the joy?",
        "Awesome! Can you share why you're happy?",
        "That's fantastic! What's making you smile?"
    ],
    "stress":[
        "Stress can be hard. Try deep breaths or a short walk.",
        "Take a moment to relax. Want to talk about it?",
        "Feeling stressed is normal. How about some tips?",
        "Let's focus on small steps to reduce stress.",
        "Stress happens. I'm here to help you talk through it."
    ],
    "anxious":[
        "Feeling anxious is normal. Want to share what's worrying you?",
        "Take a deep breath. What's making you anxious?",
        "Anxiety can be heavy. I'm listening.",
        "You’re not alone. Do you want to talk about it?",
        "It's okay to feel nervous. Can you tell me more?"
    ],
    "angry":[
        "Anger is natural. Want to vent a little?",
        "Take a deep breath. Can you tell me why you're angry?",
        "It's okay to be frustrated. I'm here to listen.",
        "I hear you. Let's try to calm down together.",
        "Anger can be tough. Want to talk about it?"
    ],
    "tired":[
        "Rest is important. Are you getting enough sleep?",
        "Sounds like you need a break. Try to relax.",
        "Being tired can affect everything. Take care of yourself.",
        "Make sure to recharge. How tired are you feeling?",
        "Try to rest and relax for a bit."
    ],
    "greeting":[
        "Hi there! How are you feeling today?",
        "Hello! Nice to meet you. How's your day going?",
        "Hey! Hope you're doing well. How are you?",
        "Hi! How's everything going today?",
        "Hello! I'm here to chat if you want."
    ],
    "smalltalk":[
        "I'm doing well, thanks! How about you?",
        "All good here. How are you feeling today?",
        "I'm fine! What about you?",
        "Doing great! How’s your day going?",
        "I'm good! Do you want to talk about your feelings?"
    ],
    "gratitude":[
        "You're welcome! I'm here to help.",
        "No problem! Glad I can chat with you.",
        "Anytime! How are you feeling now?",
        "Happy to help! Do you want to talk more?",
        "You're welcome! Let's keep chatting if you want."
    ],
    "affirmation":[
        "Great!", "Awesome!", "Good to hear!", "Nice!", "Fantastic!"
    ],
    "negation":[
        "That's okay.", "No worries.", "Understood.", "Alright.", "Okay."
    ],
    "help":[
        "Sure, tell me what's going on. How can I help?",
        "I'm here to help. What's troubling you?",
        "Let's work through it together. What do you need?",
        "I can give tips or just listen. What would you like?",
        "Don't worry, I'm listening. How can I support you?"
    ],
    "fun":[
        "Haha, that's funny!", "LOL!", "That's interesting!", "Wow!", "Yikes!"
    ],
    "sad_anxious":[
        "It seems you're feeling both sad and anxious. Take a deep breath. Want to talk about it?",
        "Sadness and anxiety together can feel overwhelming. I'm here for you.",
        "You might be feeling heavy emotions. Do you want to share?",
        "Feeling both sad and anxious is normal. I'm listening.",
        "I understand it can be tough. Want to talk?"
    ],
    "happy_stress":[
        "Even when stressed, it's nice to have happy moments. Focus on what makes you happy!",
        "Happy moments can balance stress. What's one good thing today?",
        "It's okay to feel happy even under pressure. Want to share?",
        "Even in stress, finding joy matters. What made you happy?",
        "Balancing stress with happiness is great. Tell me more!"
    ],
    "default":[
        "I see. Can you tell me more?",
        "I'm listening. Please share more.",
        "Interesting. Can you explain a bit more?",
        "Hmm, tell me more about that.",
        "I want to understand better. Can you elaborate?"
    ]
}

# Dual emotion detection
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

    # Check dual emotions first
    dual_response = detect_dual_emotion(user_input)
    if dual_response:
        print("MindBuddy:", dual_response)
        continue

    # Predict with ML
    X_input = vectorizer.transform([user_input])
    probs = model.predict_proba(X_input)[0]
    predicted_emotion = model.classes_[probs.argmax()]

    # Use default if model is unsure
    if probs.max() < 0.5:
        response = random.choice(responses["default"])
    else:
        response = random.choice(responses.get(predicted_emotion, responses["default"]))

    print("MindBuddy:", response)
