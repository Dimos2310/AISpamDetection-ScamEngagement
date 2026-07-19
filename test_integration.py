import sys
from pathlib import Path
from script import classify_email

def main():
    print("="*60)
    print("INTEGRATION TEST: Local SVM Email Classifier")
    print("="*60)

    # Test Case 1: Legitimate Email
    legit_email = (
        "Subject: Project Status Update\n\n"
        "Body:\n"
        "Hi John,\n\n"
        "I hope you are doing well. Just wanted to let you know that we completed the main "
        "tasks for this week. Please review the updated documentation when you get a chance.\n\n"
        "Thanks,\n"
        "Emily"
    )

    # Test Case 2: Nigerian Prince Scam Email (SPAM)
    spam_email = (
        "Subject: URGENT ASSISTANCE REQUIRED\n\n"
        "Body:\n"
        "DEAR FRIEND,\n\n"
        "I am Prince Abakaliki of the Central Bank of Nigeria. I am writing to request your urgent "
        "assistance in transferring 25 MILLION DOLLARS from a dormant account. Due to government regulations, "
        "this transaction must be kept strictly confidential. Please provide your full bank account details "
        "and phone number immediately. I will pay you 40% of the funds as a fee.\n\n"
        "Yours faithfully,\n"
        "Prince Abakaliki"
    )

    # Test Case 3: Greek Scam Email (SPAM)
    greek_spam_email = (
        "Subject: Επείγουσα Ειδοποίηση: Μεταφορά Κεφαλαίων\n\n"
        "Body:\n"
        "Αγαπητέ φίλε,\n\n"
        "Είμαι ο δικηγόρος Marcus Vance και σας στέλνω αυτό το μήνυμα εκ μέρους του πελάτη μου που απεβίωσε. "
        "Έχετε κληρονομήσει το ποσό των 10.5 εκατομμυρίων ευρώ. Για να ξεκινήσει η διαδικασία μεταφοράς της κληρονομιάς, "
        "χρειαζόμαστε τα τραπεζικά σας στοιχεία, το ονοματεπώνυμο και το τηλέφωνό σας άμεσα. "
        "Παρακαλώ απαντήστε επειγόντως."
    )

    print("\n--- Test Case 1: Legitimate Email ---")
    result1 = classify_email(legit_email)
    print("Classification:", result1.get("classification"))
    print("Confidence:", result1.get("confidence"))
    print("Reason:", result1.get("reason"))

    print("\n--- Test Case 2: English Scam Email ---")
    result2 = classify_email(spam_email)
    print("Classification:", result2.get("classification"))
    print("Confidence:", result2.get("confidence"))
    print("Reason:", result2.get("reason"))

    print("\n--- Test Case 3: Greek Scam Email ---")
    result3 = classify_email(greek_spam_email)
    print("Classification:", result3.get("classification"))
    print("Confidence:", result3.get("confidence"))
    print("Reason:", result3.get("reason"))

    print("\n" + "="*60)
    if (result1.get("classification") == "LEGITIMATE" and 
        result2.get("classification") == "SPAM" and 
        result3.get("classification") == "SPAM"):
        print("TEST SUCCESS: Classifier working correctly in both English and Greek!")
    else:
        print("TEST FAILURE: Check classification results.")
    print("="*60)

if __name__ == "__main__":
    main()
