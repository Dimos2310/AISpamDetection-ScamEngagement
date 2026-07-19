import imaplib
import smtplib
import email
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import re
import os
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Dynamically append Q2 directory to python import path so we can import the feature extraction script
sys.path.append(str(Path(__file__).parent / "Q2"))
try:
    from preprocess_question2 import extract_email_features, FEATURE_COLUMNS
    print("[*] Successfully imported preprocessing pipeline from Q2/preprocess_question2.py.")
except ImportError:
    print("[!] Warning: Could not import preprocessing pipeline from Q2/preprocess_question2.py.")
    extract_email_features = None
    FEATURE_COLUMNS = None

# Load pre-trained SVM model and scaler if they exist
CLASSIFIER_MODEL = None
FEATURE_SCALER = None

model_path = Path(__file__).parent / "spam_model.pkl"
scaler_path = Path(__file__).parent / "scaler.pkl"

if model_path.exists() and scaler_path.exists():
    try:
        with open(model_path, "rb") as f:
            CLASSIFIER_MODEL = pickle.load(f)
        with open(scaler_path, "rb") as f:
            FEATURE_SCALER = pickle.load(f)
        print(f"[*] Loaded local ML model ({model_path.name}) and scaler ({scaler_path.name}) successfully.")
    except Exception as e:
        print(f"[!] Error loading ML model/scaler: {e}")
else:
    print("[!] Local ML model files not found. Will fall back to Groq API for classification.")

# ==============================================================================
# 1. SETTINGS
# ==============================================================================


SMTP_SERVER   = "smtp.gmail.com"
SMTP_PORT     = 587

# Load credentials from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
EMAIL_ACCOUNT = os.getenv("EMAIL_ACCOUNT", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")

# Verify settings and print warnings if missing
if not GROQ_API_KEY:
    print("[!] Warning: GROQ_API_KEY environment variable is not set. AI functions will fail.")
if not EMAIL_ACCOUNT or not EMAIL_PASSWORD:
    print("[!] Warning: EMAIL_ACCOUNT or EMAIL_PASSWORD environment variables are not set. Email monitoring will fail.")

IMAP_SERVER = "imap.gmail.com" # e.g. imap.gmail.com or mail.yourdomain.com
SPAM_FOLDER_NAME = '"[Gmail]/&A5EDvQO1A8ADuQO4A80DvAO3A8QDsQ-"' # Spam folder for Greek Gmail (Modified UTF-7)
HISTORY_FILE = "scambait_history.json"

# TESTS: Set a specific sender to limit processing during testing.
# If set, the script only handles emails from that address.
# For production, set to None or an empty string "".
TEST_SENDER_FILTER = ""

# ==============================================================================
# 2. AI FUNCTIONS
# ==============================================================================

def normalize_subject(subject: str) -> str:
    """
    Normalize the email subject: lower-case it, remove Re:/Fwd:/etc.,
    and remove extra whitespace.
    """
    subj = subject.lower().strip()
    # Remove prefixes like Re:, Fwd:, Fw:, Aw:, etc. repeatedly
    while True:
        old_subj = subj
        subj = re.sub(r'^(re|fwd|fw|aw|vs|reply|απ):\s*', '', subj)
        if subj == old_subj:
            break
    return " ".join(subj.split())


def strip_quoted_reply(body: str) -> str:
    """
    Remove the quoted reply section from an email body.
    Gmail typically adds lines like 'On ... wrote:' followed by lines starting with '>'.
    This function strips everything from the 'On ... wrote:' marker onwards.
    """
    # Match 'On <date> ... wrote:' pattern (Gmail/Outlook style)
    pattern = r'\r?\n\s*On\s+.{10,80}\s+wrote:\s*$'
    match = re.search(pattern, body, flags=re.MULTILINE | re.IGNORECASE)
    if match:
        cleaned = body[:match.start()].strip()
        if cleaned:
            return cleaned
    
    # Fallback: remove lines starting with '>' (generic quoting)
    lines = body.splitlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith('>'):
            break
        new_lines.append(line)
    cleaned = '\n'.join(new_lines).strip()
    return cleaned if cleaned else body.strip()

def load_history() -> dict:
    """Load the scambait conversation history."""
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"    [!] Error loading history: {e}")
        return {}


def save_history(history: dict):
    """Save the scambait conversation history."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"    [!] Error saving history: {e}")


def is_english_text(text: str) -> bool:
    """
    Detect if the email body is primarily written in English.
    Returns False if Greek characters are found or if the density of English 
    stopwords is extremely low (indicating another non-English language).
    """
    text_lower = text.lower()
    
    # Check for Greek characters (Unicode range 0370 to 03FF and 1F00 to 1FFF)
    contains_greek = bool(re.search(r'[\u0370-\u03ff\u1f00-\u1fff]', text_lower))
    if contains_greek:
        return False
        
    # Check for density of common English stopwords
    words = re.findall(r'\b[a-z]{2,}\b', text_lower)
    if not words:
        return False
        
    english_stopwords = {
        "the", "and", "to", "of", "a", "in", "is", "you", "that", "it", "he", "was", "for", "on", "are", "as", "with", "his", "they", "i",
        "this", "have", "from", "at", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we", "when", "your", "can", "said",
        "there", "use", "an", "each", "which", "she", "do", "how", "their", "if", "will", "up", "other", "about", "out", "many", "then", "them",
        "these", "so", "some", "her", "would", "make", "like", "him", "into", "time", "has", "look", "two", "more", "write", "go", "see"
    }
    
    english_word_count = sum(1 for w in words if w in english_stopwords)
    
    if len(words) >= 5:
        ratio = english_word_count / len(words)
        if ratio < 0.08:
            return False
            
    return True


def classify_email(email_text: str) -> dict:
    """
    Classify the email as SPAM or LEGITIMATE.
    First detects if the email is in English. If it is English, it tries to use
    the local SVM model. If it is in another language, or if the model files are
    not found, or if classification fails, falls back to Groq AI.
    Returns a dict with classification, confidence, and reason.
    """
    # Parse the email body text out of the input parameter
    body = email_text
    if "\n\nBody:\n" in email_text:
        body = email_text.split("\n\nBody:\n", 1)[1]

    # Check if the email text is in English
    is_eng = is_english_text(body)

    # Try using the local ML classifier first if it is available and text is English
    if is_eng and CLASSIFIER_MODEL is not None and FEATURE_SCALER is not None and extract_email_features is not None:
        try:
            # Extract features using Q2 preprocessing pipeline
            features_dict = extract_email_features(body)
            
            # Form the feature vector aligned with the exact columns order as a DataFrame
            X = pd.DataFrame([[features_dict[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
            
            # Normalize features using the pre-fitted scaler
            X_scaled = FEATURE_SCALER.transform(X)
            
            # Predict spam (1) or legitimate (0)
            prediction = CLASSIFIER_MODEL.predict(X_scaled)[0]
            
            # Compute classification confidence and spam probability
            if hasattr(CLASSIFIER_MODEL, "predict_proba"):
                probs = CLASSIFIER_MODEL.predict_proba(X_scaled)[0]
                confidence = f"{probs[prediction] * 100:.1f}%"
                spam_prob = float(probs[1])
            else:
                confidence = "100.0%"
                spam_prob = 1.0 if prediction == 1 else 0.0
            
            classification = "SPAM" if prediction == 1 else "LEGITIMATE"
            reason = "Classified by local SVM model trained on 40 handcrafted features from Q2."
            
            return {
                "classification": classification,
                "confidence": confidence,
                "spam_probability": spam_prob,
                "reason": reason
            }
        except Exception as e:
            print(f"    [!] Local ML classification failed: {e}. Falling back to Groq AI...")
    elif not is_eng:
        print("    [*] Non-English language detected. Local SVM model (trained on English features) cannot check if it is spam. Falling back to Groq AI...")

    # Fallback to Groq AI
    client = Groq(api_key=GROQ_API_KEY)
 
    # Improved prompt to reduce false positives for legitimate promotional emails and support other languages
    system_prompt = (
        "You are an expert AI email filtering system.\n"
        "Analyze the email and classify it as SPAM or LEGITIMATE.\n\n"
        "LEGITIMATE emails include:\n"
        "- Newsletters or promotions from known brands (Spotify, Netflix, Amazon, Google, etc.)\n"
        "- Order confirmations, shipping notifications, account alerts\n"
        "- Emails with unsubscribe links (this is a legal requirement, NOT a spam signal)\n"
        "- Marketing emails from real companies, even if promotional in tone\n\n"
        "SPAM/SCAM emails typically have:\n"
        "- Urgent requests for money, gift cards, or wire transfers\n"
        "- Promises of lottery winnings, inheritance, or unrealistic prizes\n"
        "- Requests for passwords, SSN, or credit card numbers\n"
        "- Impersonation of banks or governments asking for personal data\n"
        "- Sender domains that do not match the claimed brand\n"
        "- Poor grammar, generic greetings like 'Dear Friend', and pressure tactics\n\n"
        "Note: The email may be written in languages other than English (e.g., Greek). Evaluate it according to the same semantic rules, and write the 'reason' in English.\n\n"
        "Classify as SPAM only when confidence is above 85%. When in doubt, classify as LEGITIMATE.\n\n"
        "Respond ONLY in valid JSON, no other text:\n"
        "{\"classification\": \"SPAM\" or \"LEGITIMATE\", "
        "\"confidence\": \"0-100%\", \"reason\": \"short explanation\"}"
    )
 
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": "Email to analyze:\n\n" + email_text}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

 
 
def generate_scam_reply(sender: str, subject: str, body: str, conversation_history: list = None) -> str:
    """
    Generate a realistic automatic reply to the scam email for scambaiting.
    The goal is to waste the scammer's time without revealing real personal data.
    Supports multi-turn conversations by passing conversation_history.
    """
    client = Groq(api_key=GROQ_API_KEY)
 
    system_prompt = (
        "You are a witty but professional email assistant.\n"
        "A scam/spam email has been received. Write an automatic reply that appears genuine\n"
        "and interested, to waste the scammer's time (this is called 'scambaiting').\n\n"
        "Rules:\n"
        "- Sound like a real, enthusiastic person who is very interested\n"
        "- Ask for many clarifying details (full name, address, phone, company, reference number)\n"
        "  to waste their time, but NEVER provide any real personal information\n"
        "- Be polite and friendly in tone\n"
        "- Keep it between 100-200 words\n"
        "- Write in the same language as the original email\n"
        "- Do NOT reveal that you know it is a scam\n"
        "- Sign off as 'Alex Johnson'\n\n"
        "Respond with ONLY the email body text. No subject line. No JSON."
    )

    if conversation_history:
        # Multi-turn flow: system prompt + entire chat log
        # Clean history items to contain only 'role' and 'content' fields as expected by Groq API
        # and skip draft messages that are still pending approval (not sent yet)
        cleaned_history = []
        for msg in conversation_history:
            if msg.get("role") == "assistant" and msg.get("status") == "pending_approval":
                continue
            cleaned_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        messages = [{"role": "system", "content": system_prompt}] + cleaned_history
    else:
        # First-turn flow
        user_prompt = (
            "Original scam email:\n"
            "From: " + sender + "\n"
            "Subject: " + subject + "\n\n"
            "Body:\n" + body[:1500] + "\n\n"
            "Write the reply now:"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
 
    try:
        response = client.chat.completions.create(
            messages=messages,
            # FIX: llama-3.1-70b-versatile is deprecated.
            # Replaced with llama-3.3-70b-versatile
            model="llama-3.3-70b-versatile",
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "[Error generating reply: " + str(e) + "]"
 
 
# ==============================================================================
# 3. EMAIL UTILITIES
# ==============================================================================
 
def extract_text_from_email(msg) -> str:
    """
    Extract plain text from an email.
    If plain text is not available, decode and clean HTML-only content.
    """
    plain_text = ""
    html_text = ""
    
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                try:
                    plain_text += part.get_payload(decode=True).decode(errors="replace")
                except Exception:
                    pass
            elif content_type == "text/html":
                try:
                    html_text += part.get_payload(decode=True).decode(errors="replace")
                except Exception:
                    pass
    else:
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            try:
                plain_text = msg.get_payload(decode=True).decode(errors="replace")
            except Exception:
                pass
        elif content_type == "text/html":
            try:
                html_text = msg.get_payload(decode=True).decode(errors="replace")
            except Exception:
                pass
                
    if plain_text.strip():
        return plain_text.strip()
        
    if html_text.strip():
        # Clean HTML tags using regex
        # 1. Replace <br> and </p> with newlines to keep structure
        text = re.sub(r'<br\s*/?>', '\n', html_text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)
        # 2. Remove style and script blocks completely
        text = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', text, flags=re.IGNORECASE)
        # 3. Remove all remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # 4. Decode HTML entities (like &nbsp;, &amp;, &lt;, &gt;)
        import html
        text = html.unescape(text)
        return "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        
    return ""
 
 
def send_reply(to_address: str, original_subject: str, reply_body: str) -> bool:
    """Send the automatic reply via SMTP using STARTTLS."""
    msg = MIMEMultipart("alternative")
    msg["From"]    = EMAIL_ACCOUNT
    msg["To"]      = to_address
    msg["Subject"] = "Re: " + original_subject
    msg.attach(MIMEText(reply_body, "plain", "utf-8"))
 
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ACCOUNT, to_address, msg.as_string())
        return True
    except Exception as e:
        print("    [!] Failed to send email: " + str(e))
        return False
 
 
def decode_subject(msg) -> str:
    """Decode the Subject header safely."""
    raw, enc = decode_header(msg.get("Subject", "(no subject)"))[0]
    if isinstance(raw, bytes):
        return raw.decode(enc or "utf-8", errors="replace")
    return raw or "(no subject)"
 
 
def extract_reply_address(msg) -> str:
    """Return Reply-To if present, otherwise return From."""
    addr = msg.get("Reply-To") or msg.get("From") or ""
    if "<" in addr and ">" in addr:
        addr = addr.split("<")[1].rstrip(">").strip()
    return addr
 
 
def detect_spam_folder(mail) -> str:
    """Dynamically detects the Spam folder name on the IMAP server."""
    status, folder_list = mail.list()
    if status != "OK":
        return SPAM_FOLDER_NAME
    
    for folder_info in folder_list:
        if not folder_info:
            continue
        try:
            info_str = folder_info.decode("utf-8", errors="replace").lower()
            # Parse folder name
            parts = re.findall(r'"([^"]*)"', info_str)
            folder_name = parts[-1] if len(parts) >= 2 else info_str.split()[-1]
            folder_name = folder_name.strip('"').strip()
            
            # Check for \Spam attribute or keywords
            if "\\spam" in info_str or "spam" in folder_name.lower() or "junk" in folder_name.lower() or "ανεπιθύμητα" in folder_name.lower() or "&a5edvqo1a8aduqo4a80dvo3a8qdsq-" in folder_name.lower():
                # Preserve case by retrieving it from raw decoded string
                orig_str = folder_info.decode("utf-8", errors="replace")
                orig_parts = re.findall(r'"([^"]*)"', orig_str)
                actual_name = orig_parts[-1] if len(orig_parts) >= 2 else orig_str.split()[-1]
                return f'"{actual_name.strip()}"'
        except Exception:
            pass
    return SPAM_FOLDER_NAME


def get_all_folders(mail) -> list:
    """Gets all available email folders from the IMAP server, excluding outgoing/deleted folders."""
    folders = []
    status, folder_list = mail.list()
    if status != "OK":
        return ["INBOX"]
    
    for folder_info in folder_list:
        if not folder_info:
            continue
        try:
            info_str = folder_info.decode("utf-8", errors="replace")
            # Parse the folder name (usually the last quoted or unquoted string)
            parts = re.findall(r'"([^"]*)"', info_str)
            if len(parts) >= 2:
                folder_name = parts[-1]
            else:
                tokens = info_str.split()
                if tokens:
                    folder_name = tokens[-1]
            
            folder_name = folder_name.strip('"').strip()
            if folder_name:
                folders.append(folder_name)
        except Exception as e:
            print(f"    [!] Error parsing folder name: {e}")
            
    safe_folders = []
    for f in folders:
        f_lower = f.lower()
        
        # Always include the user-configured spam folder
        is_spam_folder = (f.lower() == SPAM_FOLDER_NAME.strip('"').lower()) or ("spam" in f_lower and "sent" not in f_lower and "draft" not in f_lower)
        if is_spam_folder:
            safe_folders.append(f)
            continue
            
        if any(keyword in f_lower for keyword in ["sent", "draft", "trash", "bin", "deleted", "απεσταλμένα", "πρόχειρα", "κάδος", "διαγραμμένα"]):
            continue
            
        safe_folders.append(f)
        
    seen = set()
    unique_folders = []
    for f in safe_folders:
        if f not in seen:
            seen.add(f)
            unique_folders.append(f)
            
    # Force 'INBOX' to be the first checked folder
    clean_unique = []
    for f in unique_folders:
        if f.lower() != "inbox":
            clean_unique.append(f)
    clean_unique.insert(0, "INBOX")
    
    return clean_unique


def get_imap_date_48h_ago() -> str:
    """Calculates the date 48 hours ago and formats it for IMAP (locale-independent)."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    target_date = datetime.now() - timedelta(days=2)
    day = target_date.day
    month_name = months[target_date.month - 1]
    year = target_date.year
    return f"{day:02d}-{month_name}-{year}"


# ==============================================================================
# 4. MAIN LOGIC
# ==============================================================================

def process_inbox():
    global SPAM_FOLDER_NAME
    print("=" * 52)
    print("   Email Scam Auto-Reply System")
    print("=" * 52)
    print("\nConnecting to " + IMAP_SERVER + "...")
 
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
        print("Connected successfully!\n")
    except Exception as e:
        print("Connection failed: " + str(e))
        return
 
    # Dynamically detect the spam folder of the user's account (English or Greek)
    SPAM_FOLDER_NAME = detect_spam_folder(mail)
    
    folders = get_all_folders(mail)
    print(f"Found folders to check: {', '.join(folders)}")
    
    imap_date = get_imap_date_48h_ago()
    print(f"Filtering: Only checking unread emails received since {imap_date} (last 48 hours).\n")
 
    stats = {"total": 0, "spam": 0, "legit": 0, "replied": 0, "skipped": 0}
    history = load_history()
 
    for folder in folders:
        print(f"Checking folder: '{folder}'...")
        status, _ = mail.select(f'"{folder}"') 
        if status != "OK":
            print(f"  Could not select folder: '{folder}'. Skipping.")
            continue
 
        status, messages = mail.search(None, f"UNSEEN SINCE {imap_date}")
        if status != "OK":
            print(f"  Could not search unseen emails in '{folder}' with date filter. Skipping.")
            mail.close()
            continue

        email_ids = messages[0].split()
        if not email_ids:
            mail.close()
            continue
            
        print(f"  Found {len(email_ids)} unread emails in '{folder}'.\n")
 
        for e_id in email_ids:
            stats["total"] += 1
            status, msg_data = mail.fetch(e_id, "(RFC822)")
            if status != "OK" or not msg_data:
                continue
 
            for part in msg_data:
                if not isinstance(part, tuple):
                    continue
 
                msg     = email.message_from_bytes(part[1])
                subject = decode_subject(msg)
                sender  = msg.get("From", "Unknown")
                reply_addr = extract_reply_address(msg).strip().lower() or sender.strip().lower()
 
                # Test Sender Filter Check
                if TEST_SENDER_FILTER and reply_addr != TEST_SENDER_FILTER.strip().lower():
                    print(f"Skipping email from non-test sender: {reply_addr}")
                    continue

                print("-" * 52)
                print("From:    " + sender)
                print("Subject: " + subject)
                print("Folder:  " + folder)
 
                body = extract_text_from_email(msg)
                if not body:
                    print("Empty or HTML-only email. Skipping.")
                    stats["skipped"] += 1
                    continue
 
                # Compute the conversation key for scambaiting history
                norm_subj = normalize_subject(subject)
                conv_key = f"{reply_addr} | {norm_subj}"
                is_ongoing_scambait = conv_key in history

                if is_ongoing_scambait:
                    print(f"\n[Multi-turn] Found active scambaiting session for: {conv_key}")
                    classification = "SPAM"
                    confidence = "100% (Ongoing)"
                    scam_prob = 1.0
                    reason = "Existing active scambaiting conversation thread."
                else:
                    # STEP 1: Classification
                    ai_input = "Subject: " + subject + "\n\nBody:\n" + body[:2000]
                    result   = classify_email(ai_input)
                    classification = result.get("classification", "UNKNOWN")
                    confidence     = result.get("confidence", "N/A")
                    scam_prob      = result.get("spam_probability", 1.0)
                    reason         = result.get("reason", "N/A")
 
                print("AI Decision: " + classification + " (Confidence: " + confidence + ")")
                print("Reason:      " + reason)
 
                if classification == "SPAM":
                    stats["spam"] += 1
 
                    # STEP 2: Generate reply
                    print("\nGenerating automatic reply draft...")
                    if is_ongoing_scambait:
                        # Append the scammer's new reply to the existing history list
                        # Strip quoted reply text so only the new message is stored
                        clean_body = strip_quoted_reply(body[:2000])
                        history[conv_key].append({
                            "role": "user", 
                            "content": clean_body,
                            "scam_probability": scam_prob
                        })
                        reply = generate_scam_reply(sender, subject, body, conversation_history=history[conv_key])
                    else:
                        # Initial reply
                        reply = generate_scam_reply(sender, subject, body)
                        # Initialize history with original message
                        history[conv_key] = [
                            {
                                "role": "user", 
                                "content": f"Subject: {subject}\n\nBody:\n{body[:2000]}",
                                "scam_probability": scam_prob
                            },
                        ]
                    
                    # Append our reply to history (with pending approval status)
                    history[conv_key].append({
                        "role": "assistant", 
                        "content": reply,
                        "status": "pending_approval"
                    })
                    save_history(history)
 
                    print("\n--- REPLY DRAFT SAVED FOR APPROVAL ---")
                    print("To: " + reply_addr)
                    print("Draft Preview:")
                    print(reply)
                    print("--------------------------------------\n")
 
                    # STEP 4: Archive/Move the spam email (to prevent processing it again)
                    clean_spam_folder = SPAM_FOLDER_NAME.strip('"').lower()
                    clean_current_folder = folder.strip('"').lower()
                    if clean_current_folder != clean_spam_folder:
                        mail.copy(e_id, SPAM_FOLDER_NAME)
                        mail.store(e_id, "+FLAGS", "\\Deleted")
                        print(f"Moved email to '{SPAM_FOLDER_NAME}' folder.")
                    else:
                        mail.store(e_id, "+FLAGS", "\\Seen")
                        print("Marked spam email as read.")
 
                else:
                    stats["legit"] += 1
                    print("Legitimate email — no action taken.")
 
        mail.close()
        
    mail.logout()
 
    print("\n" + "=" * 52)
    print("  SUMMARY")
    print("=" * 52)
    print("  Total emails:      " + str(stats["total"]))
    print("  Spam/Scam:         " + str(stats["spam"]))
    print("  Legitimate:        " + str(stats["legit"]))
    print("  Automatic replies: " + str(stats["replied"]))
    print("  Skipped:           " + str(stats["skipped"]))
    print("=" * 52)
    print("Done.")
 
 
# ==============================================================================
# 5. GUI DASHBOARD (FLASK SERVER)
# ==============================================================================
from flask import Flask, render_template, jsonify, request
import threading

app = Flask(__name__, template_folder='templates')
is_checking = False
checking_lock = threading.Lock()

@app.route('/')
def home():
    """Serve the Web Telegram-like UI."""
    return render_template('gui.html')

@app.route('/api/history')
def api_history():
    """API endpoint to get scambait history."""
    return jsonify(load_history())

@app.route('/api/check_status', methods=['GET'])
def check_status():
    """API endpoint to get email checking status."""
    global is_checking
    return jsonify({"is_checking": is_checking})

@app.route('/api/check_mail', methods=['POST'])
def check_mail():
    """Trigger a live check of the email box in a background thread."""
    global is_checking
    
    with checking_lock:
        if is_checking:
            return jsonify({"status": "already_checking"})
        is_checking = True

    def run_mail_check():
        global is_checking
        try:
            print("\n" + "="*50)
            print("[*] STARTING LIVE EMAIL CHECK IN BACKGROUND THREAD...")
            print("="*50)
            process_inbox()
        except Exception as err:
            print(f"[!] Background email check error: {err}")
        finally:
            with checking_lock:
                is_checking = False
            print("\n" + "="*50)
            print("[*] LIVE EMAIL CHECK THREAD COMPLETED.")
            print("="*50)

    # Spawn thread to avoid blocking Flask
    threading.Thread(target=run_mail_check, daemon=True).start()
    return jsonify({"status": "started"})


@app.route('/api/approve_reply', methods=['POST'])
def approve_reply():
    """Approve a draft reply and send it via SMTP."""
    data = request.json or {}
    key = data.get("key")
    if not key:
        return jsonify({"status": "error", "message": "Missing key"}), 400
        
    history = load_history()
    if key not in history:
        return jsonify({"status": "error", "message": "Conversation not found"}), 404
        
    thread = history[key]
    
    # Find the last message which should be the pending reply from assistant
    pending_idx = -1
    for idx, msg in enumerate(thread):
        if msg.get("role") == "assistant" and msg.get("status") == "pending_approval":
            pending_idx = idx
            break
            
    if pending_idx == -1:
        return jsonify({"status": "error", "message": "No pending draft reply found"}), 404
        
    reply_body = thread[pending_idx]["content"]
    
    # Parse sender email and subject from key
    try:
        parts = key.split(" | ", 1)
        reply_addr = parts[0].strip()
        subject = parts[1].strip()
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to parse key: {e}"}), 400
        
    # Send SMTP reply
    print(f"[*] SMTP: Sending approved reply to: {reply_addr}")
    if send_reply(reply_addr, subject, reply_body):
        # Update status of this message
        thread[pending_idx]["status"] = "sent"
        save_history(history)
        print("[*] SMTP: Approved reply sent successfully.")
        return jsonify({"status": "approved_and_sent"})
    else:
        return jsonify({"status": "error", "message": "SMTP transmission failed"}), 500


@app.route('/api/reject_reply', methods=['POST'])
def reject_reply():
    """Reject a draft reply, deleting it, and optionally deleting the entire thread if requested."""
    data = request.json or {}
    key = data.get("key")
    delete_thread = data.get("delete_thread", False)
    
    if not key:
        return jsonify({"status": "error", "message": "Missing key"}), 400
        
    history = load_history()
    if key not in history:
        return jsonify({"status": "error", "message": "Conversation not found"}), 404
        
    if delete_thread:
        # User wants to delete the whole conversation (e.g. if it was a false positive)
        print(f"[*] Reject: Deleting entire conversation for: {key}")
        if key in history:
            del history[key]
        save_history(history)
        return jsonify({"status": "thread_deleted"})
        
    # Otherwise just remove the last draft reply
    thread = history[key]
    pending_idx = -1
    for idx, msg in enumerate(thread):
        if msg.get("role") == "assistant" and msg.get("status") == "pending_approval":
            pending_idx = idx
            break
            
    if pending_idx != -1:
        print(f"[*] Reject: Deleting pending reply draft for: {key}")
        del thread[pending_idx]
        if len(thread) == 0:
            if key in history:
                del history[key]
        save_history(history)
        return jsonify({"status": "draft_deleted"})
    else:
        return jsonify({"status": "error", "message": "No pending draft reply found"}), 404



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ScamGram: Email Scambaiting Auto-Reply Client & Monitor")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode to check email once and exit immediately.")
    args = parser.parse_args()
    
    if args.cli:
        process_inbox()
    else:
        import webbrowser
        from threading import Timer
        
        print("=" * 60)
        print("      ScamGram Live GUI Dashboard & Email Monitor")
        print("=" * 60)
        print("Loading history from:", os.path.abspath(HISTORY_FILE))
        
        # Open the browser automatically 1.2 seconds after startup
        def open_browser():
            webbrowser.open_new("http://127.0.0.1:5000/")
        Timer(1.2, open_browser).start()
        
        # Run Flask server
        app.run(host='127.0.0.1', port=5000, debug=False)
 
