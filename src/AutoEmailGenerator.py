import os
import re
import time
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openai

from email.mime.text import MIMEText
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import OpenAI

# === Configure OpenAI ===
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Gmail API SCOPES ===
SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.send'
]

# === Load and preprocess classification data ===
emails = pd.read_json("Synthetic_Emails.json")

# Combine subject and body, lowercase text
emails["text"] = (emails["subject"] + " " + emails["body"]).str.lower()

# Remove duplicates to reduce leakage from synthetic templates
emails = emails.drop_duplicates(subset="text").sample(frac=1.0, random_state=42).reset_index(drop=True)

# Extract text and labels
text = emails["text"]
labels = emails["intent"]

# === Tokenization ===
vocab_size = 10000
max_length = 100
text_tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
text_tokenizer.fit_on_texts(text)
sequences = text_tokenizer.texts_to_sequences(text)
X = pad_sequences(sequences, padding='post', truncating='post', maxlen=max_length)

# === Label encoding ===
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# === Stratified train-test split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Build model ===
model = Sequential([
    Embedding(vocab_size, 128),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.2)),
    Dropout(0.4),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(loss=SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), verbose=0)
model.evaluate(X_val, y_val)

#===USE TF MODEL TO PREDICT EMAIL INTENT===
def predict_intent(subject, body):
    combined = [subject + " " + body]
    seq = text_tokenizer.texts_to_sequences(combined)
    pad = pad_sequences(seq, maxlen=50)
    pred = model.predict(pad)
    return encoder.inverse_transform([np.argmax(pred)])[0]

# ===CREATE PROMPT TO FEED TO CHAT===
def generate_prompt_from_intent(intent, subject, body):
    if intent == "refund_request":
        return f"A customer asked for a refund. Here's their message:\nSubject: {subject}\n\n{body}\nPlease write a helpful and polite response."
    elif intent == "product_question":
        return f"A customer has a product question. Message:\nSubject: {subject}\n\n{body}\nRespond with accurate product information."
    elif intent == "appointment_booking":
        return f"A customer wants to book an appointment:\nSubject: {subject}\n\n{body}\nRespond confirming their request and ask for availability."
    elif intent == "complaint":
        return f"A customer is upset or has a complaint. Message:\nSubject: {subject}\n\n{body}\nPlease write a respectful, empathetic response apologizing and offering help."
    else:
        return f"Here's a general customer message:\nSubject: {subject}\n\n{body}\nWrite a professional and polite response."


# ===ASK GPT API===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def generate_reply_with_gpt(prompt_text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful customer service assistant."},
            {"role": "user", "content": prompt_text}
        ],
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

#=== AUTHORIZE CRED AND TOKENS===
def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            print("Granted scopes:", creds.scopes)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

# ===CREATE MESSAGE (in base 64)===
def create_message(sender, to, subject, message_text):
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': raw}

# ===SEND MESSAGE FUNCTION===
def send_message(service, user_id, message):
    try:
        sent_message = service.users().messages().send(userId=user_id, body=message).execute()
        print(f"Message sent! ID: {sent_message['id']}")
    except HttpError as error:
        print(f"An error occurred while sending: {error}")


# ===MAIN LOOP===
def check_and_reply():
    service = get_gmail_service()
    try:
        # Ensure AutoReplied label exists
        labels_response = service.users().labels().list(userId='me').execute()
        label_names = [label['name'] for label in labels_response['labels']]
        if 'AutoReplied' not in label_names:
            service.users().labels().create(userId='me', body={'name': 'AutoReplied'}).execute()
        labels_response = service.users().labels().list(userId='me').execute()
        label_id = next((label['id'] for label in labels_response['labels'] if label['name'] == 'AutoReplied'), None)

        query = "is:unread -label:AutoReplied"
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], q=query).execute()
        messages = results.get('messages', [])

        if not messages:
            print("No new messages to reply to.")
            return

        for msg in messages:
            msg_id = msg['id']
            message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
            headers = message['payload']['headers']
            parts = message['payload'].get('parts', [])

            sender = next((h['value'] for h in headers if h['name'] == 'From'), None)
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "(No subject)")

            # Skip no-reply senders
            if sender and re.search(r'(no-?reply|@google\.com|@accounts\.google\.com)', sender, re.IGNORECASE):
                print(f"Skipping no-reply sender: {sender}")
                continue

            # Extract plain text email body
            body = ""
            for part in parts:
                if part['mimeType'] == 'text/plain':
                    body = base64.urlsafe_b64decode(part['body']['data']).decode()
                    break

            print(f"Replying to: {sender} | Subject: {subject}")

            intent = predict_intent(subject, body)
            print(f"Predicted intent: {intent}")

            prompt = generate_prompt_from_intent(intent, subject, body)
            reply_text = generate_reply_with_gpt(prompt)

            reply = create_message("me", sender, f"Re: {subject}", reply_text)
            send_message(service, "me", reply)

            service.users().messages().modify(
                userId='me',
                id=msg_id,
                body={'removeLabelIds': ['UNREAD'], 'addLabelIds': [label_id]}
            ).execute()

    except HttpError as error:
        print(f"An error occurred: {error}")

# === Auto-run every 30 seconds ===
print("Running SmartTextAssistant... Press Ctrl+C to stop.")
while True:
    check_and_reply()
    time.sleep(30)
