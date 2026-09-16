# ScamGram: AI-Driven Email Spam Detection & Active Deception (Scambaiting)

> **MSc Semester Project 1**  
> **Course:** Advanced Cybersecurity Topics & AI  
> **Institution:** University of Piraeus | School of Information and Communication Technologies | Department of Digital Systems  
> **Program:** MSc Digital Systems Security  
> **Supervisor:** Prof. Apostolis Zarras  
> **Date:** June 2026  

---

## 👨‍🎓 Project Team Members

| Name & Surname | University E-mail | Student ID |
| :--- | :--- | :--- |
| **Miltiadis Desypris** | `miltiadis_desipris@ssl-unipi.gr` | `mte25009` |
| **Maria Eleni Kasteli** | `m_kasteli@ssl-unipi.gr` | `mte25020` |
| **Dimosthenis Kyriakidis** | `dimosthenis_kiriakidis@ssl-unipi.gr` | `mte25024` |
| **Panagiotis Lampiris** | `panagiotis_lampiris@ssl-unipi.gr` | `mte25028` |
| **Panagiotis Nektarios Tzortzinis** | `panagiotis-nektarios_tzortzinis@ssl-unipi.gr` | `mte25048` |
| **Angelos Triantafyllou** | `aggelos_triantafillou@ssl-unipi.gr` | `mte25050` |

---

## 📌 Project Overview

ScamGram is a hybrid cybersecurity tool that shifts email defense from a passive perimeter filter to an **active, adversarial engagement system**. Developed for the MSc in Digital Systems Security at the University of Piraeus, it combines local Machine Learning (SVM), Generative AI (Groq Llama 3 models), standard email protocols (IMAP & SMTP), and an interactive Flask web dashboard.

---

## Key Features

1. **Dual-Layer Email Classification**:
   - **Local SVM Classifier**: A Support Vector Machine (SVM) model with an RBF kernel, locally classifying English emails based on 39 handcrafted linguistic, structural, and semantic features (e.g., urgency terms, bank detail requests, fee indicators).
   - **Generative AI Fallback**: If local model files are absent, or a non-English language (such as Greek) is detected, the system automatically routes the email text to **Llama 3.1** (via the Groq API) for high-accuracy zero-shot classification and multilingual support.
2. **Active Adversarial Deception (Scambaiting)**:
   - When a scam email (such as an advance-fee scam, lottery warning, or cryptocurrency phishing email) is intercepted, the system queries **Llama 3.3** via Groq to craft an engaging, context-aware reply under the fictional persona *Alex Johnson*.
   - The generated response is designed to sound enthusiastic and gullible, asking for extensive clarifying details to waste the scammer's operational time and resources, without exposing any real personal data.
3. **Human-in-the-Loop Web Interface**:
   - A Telegram-style dashboard built with Flask allows the operator to monitor checked folders, view live conversation histories, and inspect generated draft replies.
   - No emails are ever dispatched automatically; the operator must explicitly click **Approve** or **Reject** on the dashboard before replies are sent via SMTP.
4. **Automatic Inbox Archiving**:
   - Once classified as spam/scam, emails are automatically transferred out of the user's active Inbox to a dedicated Spam folder to avoid repeat checks and clutter.

---

## Directory Structure

```text
├── Q2/                             # Feature extraction & preprocessing
│   ├── datasets/                   # Raw CSV datasets folder (~2,500 emails)
│   ├── preprocess_question2.py     # Preprocessing pipeline (extracts 39 features)
│   └── training_dataset.csv        # Preprocessed CSV with extracted feature rows
├── templates/                      # Flask UI templates
│   └── gui.html                    # Telegram-style web dashboard interface
├── TestingNewImpl/                 # Jupyter notebook testing environment
├── .env.example                    # Template for environment configuration
├── .gitignore                      # Standard rules for Git exclusions
├── requirements.txt                # Project dependencies list
├── scaler.pkl                      # Serialized StandardScaler for SVM normalization
├── spam_model.pkl                  # Serialized trained SVM classifier model
├── script.py                       # Main application entrypoint (Flask + mail client)
├── test_integration.py             # Integration test script for checking the classifier
└── train_classifier.py             # Script to fit & export the local SVM model
```

---

## System Architecture

The email processing lifecycle flows as follows:

```
[Unseen Emails (last 48h)]
          │
          ▼
   [Language Check]
     /          \
  (English)   (Greek/Other)
   /              \
  ▼                ▼
[Local RBF-SVM]   [Groq Llama 3.1]
  \                /
   ▼              ▼
[Spam or Legitimate?]
     │
     ├─► [Legitimate] ──► No action.
     │
     └─► [Spam/Scam]
           │
           ▼
[Groq Llama 3.3 Auto-Reply Draft]
           │
           ▼
[Flask Dashboard (Pending User Approval)]
     │
     ├─► [Approved] ──► Send SMTP Reply & Move to Spam Folder
     └─► [Rejected] ──► Delete Draft / Cancel Thread
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher.
- A **Groq Cloud API Key** (register at [groq.com](https://console.groq.com)).
- A **Gmail account** (or other IMAP/SMTP provider) with an **App Password** set up (Standard password will not work if 2FA is active).

> **Note on Groq model IDs:** `script.py` calls the Groq API with the model IDs `llama-3.1-8b-instant` (email classification fallback) and `llama-3.3-70b-versatile` (scambait reply generation). Groq periodically retires/renames models, so these IDs may return a `404 model_not_found` error depending on when you run the project and which models your API key currently has access to. If that happens, run `client.models.list()` (see the `groq` Python SDK) to see the models currently available to your key and update the `model=` parameters in `script.py` accordingly (lines ~264 and ~330). This does not affect the local SVM classifier path, which works independently of Groq.

### Step 1: Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-folder>
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
1. Copy the example configuration file:
   ```bash
   cp .env.example .env
   ```
2. Open the `.env` file and fill in your details:
   ```env
   GROQ_API_KEY=gsk_your_actual_groq_api_key
   EMAIL_ACCOUNT=your_email@gmail.com
   EMAIL_PASSWORD=your_app_password
   ```

---

## How to Run

### 1. Run the Integration Tests
Before starting the servers, check that your environment and classifier are functioning properly:
```bash
python test_integration.py
```

### 2. Start the Live GUI Dashboard (Flask Server)
To run ScamGram in GUI mode, execute the main script:
```bash
python script.py
```
This starts the local web server at `http://127.0.0.1:5000/` and automatically opens a browser window.
- In the dashboard, click **"Check Mail"** in the sidebar to scan your mailbox folders for unseen spam.
- View draft replies in the chat-like window, modify them if needed, and hit **"Approve"** to send them via SMTP.

### 3. CLI Mode (Single-run Check)
If you wish to scan your mail once directly in the command terminal without launching the GUI:
```bash
python script.py --cli
```

### 4. Re-training the Model (Optional)
If you wish to preprocess the raw email datasets and train the SVM classifier from scratch:
1. Place raw labeled CSV files into `Q2/datasets/`.
2. Run the preprocessing script:
   ```bash
   python Q2/preprocess_question2.py
   ```
3. Run the training script:
   ```bash
   python train_classifier.py
   ```
This updates `spam_model.pkl` and `scaler.pkl` in your project root.
