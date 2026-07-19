
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SPAM_VALUES = {
    "1", "1.0", "true", "t", "yes", "y", "spam", "scam", "phishing",
    "junk", "malicious", "fraud", "fraudulent", "positive"
}

LEGITIMATE_VALUES = {
    "0", "0.0", "false", "f", "no", "n", "ham", "legitimate", "legit",
    "not_spam", "not spam", "non-spam", "non spam", "clean", "safe",
    "normal", "negative"
}

BODY_COLUMN_CANDIDATES = [
    "body", "email_body", "email body", "message", "message_body", "text",
    "email_text", "content", "mail_body", "email", "raw_body"
]

FLAG_COLUMN_CANDIDATES = [
    "flag", "label", "spam", "is_spam", "spam_flag", "target", "class",
    "category", "type", "classification", "is_spam_flag"
]

WORD_RE = re.compile(r"\b[\w']+\b", flags=re.UNICODE)
URL_RE = re.compile(
    r"https?://\S+|www\.\S+|\bURL_TOKEN\b|<\s*URL\s*>|\[\s*URL\s*\]",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(
    r"\b[\w.-]+@[\w.-]+\.\w+\b|\bEMAIL_TOKEN\b|<\s*EMAIL\s*>|\[\s*EMAIL\s*\]",
    re.IGNORECASE,
)
PHONE_RE = re.compile(
    r"\+?\d[\d\s().-]{7,}\d|\bPHONE_TOKEN\b|<\s*PHONE\s*>|\[\s*PHONE\s*\]",
    re.IGNORECASE,
)
MONEY_RE = re.compile(
    r"\b(?:MONEY_TOKEN|AMOUNT_TOKEN|CURRENCY_TOKEN)\b|"
    r"<\s*(?:MONEY|AMOUNT|CURRENCY)\s*>|\[\s*(?:MONEY|AMOUNT|CURRENCY)\s*\]|"
    r"[$€£]\s?\d+(?:[,.]\d+)*|"
    r"\b\d+(?:[,.]\d+)*\s?(?:million|billion|thousand|usd|eur|gbp|dollars?|euros?|pounds?)\b",
    re.IGNORECASE,
)
ANON_ENTITY_RE = re.compile(
    r"\b(?:PERSON_TOKEN|NAME_TOKEN|ORG_TOKEN|ORGANIZATION_TOKEN|LOCATION_TOKEN|COUNTRY_TOKEN|DATE_TOKEN)\b|"
    r"<\s*(?:PERSON|NAME|ORG|ORGANIZATION|LOCATION|COUNTRY|DATE)\s*>|"
    r"\[\s*(?:PERSON|NAME|ORG|ORGANIZATION|LOCATION|COUNTRY|DATE)\s*\]",
    re.IGNORECASE,
)
ALL_CAPS_WORD_RE = re.compile(r"\b[A-Z]{3,}\b")

URGENCY_TERMS = [
    "urgent", "immediately", "as soon as possible", "act now", "right away",
    "without delay", "quickly", "today", "deadline", "important", "final notice"
]

SECRECY_TERMS = [
    "confidential", "secret", "private", "do not tell", "keep this", "between us",
    "discreet", "strictly confidential"
]

FINANCIAL_TERMS = [
    "money", "fund", "funds", "million", "billion", "inheritance", "lottery",
    "prize", "fortune", "transfer", "transaction", "beneficiary", "payment",
    "compensation", "donation", "investment"
]

BANK_TERMS = [
    "bank", "bank account", "account number", "swift", "iban", "routing number",
    "wire transfer", "western union", "moneygram", "atm card"
]

PERSONAL_INFO_TERMS = [
    "passport", "id card", "identity", "full name", "address", "phone number",
    "date of birth", "personal information", "occupation", "nationality", "copy of your"
]

ACTION_TERMS = [
    "send", "reply", "contact", "provide", "confirm", "fill", "complete", "click",
    "open", "respond", "forward", "call", "text me", "submit"
]

FEE_TERMS = [
    "fee", "tax", "clearance", "processing fee", "administrative fee", "customs",
    "delivery fee", "activation fee", "handling fee"
]

GREETING_TERMS = [
    "dear", "hello", "hi", "greetings", "good day", "my friend", "beloved"
]

SIGNATURE_TERMS = [
    "regards", "best regards", "yours faithfully", "sincerely", "thank you",
    "thanks", "respectfully"
]

TITLE_TERMS = [
    "mr", "mrs", "ms", "dr", "sir", "madam", "prince", "princess", "chief",
    "barrister", "attorney", "diplomat", "minister", "director", "manager", "agent"
]

COUNTRY_TERMS = [
    "nigeria", "ghana", "benin", "togo", "ivory coast", "cote d'ivoire", "south africa",
    "sierra leone", "liberia", "uk", "united kingdom", "usa", "united states", "dubai"
]

ORG_TERMS = [
    "bank", "ministry", "court", "embassy", "company", "security", "lottery", "agency",
    "department", "government", "office", "foundation", "charity"
]

FEATURE_COLUMNS = [
    "char_count",
    "word_count",
    "sentence_count",
    "paragraph_count",
    "line_count",
    "avg_word_length",
    "avg_sentence_length",
    "uppercase_ratio",
    "digit_ratio",
    "punctuation_count",
    "exclamation_count",
    "question_count",
    "multiple_exclamation_present",
    "all_caps_word_count",
    "url_count",
    "email_count",
    "phone_count",
    "money_expression_count",
    "anonymized_entity_token_count",
    "urgency_count",
    "secrecy_count",
    "financial_count",
    "bank_count",
    "personal_info_count",
    "action_request_count",
    "fee_count",
    "greeting_count",
    "signature_count",
    "title_count",
    "country_count",
    "organization_count",
    "starts_with_greeting",
    "ends_with_signature",
    "has_urgent_action_pattern",
    "has_confidentiality_pattern",
    "has_large_money_pattern",
    "has_bank_details_request_pattern",
    "has_personal_info_request_pattern",
    "has_advance_fee_pattern",
    "body_is_empty",
]

OUTPUT_BASE_COLUMNS = [
    "source_file",
    "source_row",
    "body",
    "flag",
    "flag_binary",
    "category",
]


def is_missing(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def clean_header_name(name: Any) -> str:
    return str(name or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_flag(value: Any) -> Optional[int]:
    """Convert common spam/ham labels to 1/0."""
    if is_missing(value):
        return None

    text = str(value).strip().lower()
    if text in SPAM_VALUES:
        return 1
    if text in LEGITIMATE_VALUES:
        return 0

    try:
        number = float(text)
        if number == 1.0:
            return 1
        if number == 0.0:
            return 0
    except ValueError:
        pass

    return None


def count_terms(text_lower: str, terms: Iterable[str]) -> int:
    """Count exact term/phrase occurrences using word-safe boundaries."""
    total = 0
    for term in terms:
        pattern = r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)"
        total += len(re.findall(pattern, text_lower))
    return total


def contains_any(text_lower: str, terms: Iterable[str]) -> bool:
    return count_terms(text_lower, terms) > 0


def extract_email_features(text: Any) -> Dict[str, float]:
    """Extract numeric and boolean features from one email body."""
    text = "" if text is None else str(text)
    text_lower = text.lower()
    stripped = text.strip()

    words = WORD_RE.findall(text)
    alpha_words = [word for word in words if any(ch.isalpha() for ch in word)]
    sentences = [sentence.strip() for sentence in re.split(r"[.!?]+", text) if sentence.strip()]
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    char_count = len(text)
    word_count = len(words)
    sentence_count = len(sentences)
    paragraph_count = len(paragraphs)
    line_count = len(lines)

    alpha_char_count = sum(ch.isalpha() for ch in text)
    uppercase_char_count = sum(ch.isupper() for ch in text)
    digit_count = sum(ch.isdigit() for ch in text)
    punctuation_count = sum(ch in ".,;:!?()[]{}<>/\\|@#$%^&*-_+='\"`~" for ch in text)

    avg_word_length = sum(len(word) for word in alpha_words) / max(len(alpha_words), 1)
    avg_sentence_length = word_count / max(sentence_count, 1)
    uppercase_ratio = uppercase_char_count / max(alpha_char_count, 1)
    digit_ratio = digit_count / max(char_count, 1)

    exclamation_count = text.count("!")
    question_count = text.count("?")
    multiple_exclamation_present = int(bool(re.search(r"!{2,}", text)))
    all_caps_word_count = len(ALL_CAPS_WORD_RE.findall(text))

    url_count = len(URL_RE.findall(text))
    email_count = len(EMAIL_RE.findall(text))
    phone_count = len(PHONE_RE.findall(text))
    money_expression_count = len(MONEY_RE.findall(text))
    anonymized_entity_token_count = len(ANON_ENTITY_RE.findall(text))

    urgency_count = count_terms(text_lower, URGENCY_TERMS)
    secrecy_count = count_terms(text_lower, SECRECY_TERMS)
    financial_count = count_terms(text_lower, FINANCIAL_TERMS)
    bank_count = count_terms(text_lower, BANK_TERMS)
    personal_info_count = count_terms(text_lower, PERSONAL_INFO_TERMS)
    action_request_count = count_terms(text_lower, ACTION_TERMS)
    fee_count = count_terms(text_lower, FEE_TERMS)
    greeting_count = count_terms(text_lower, GREETING_TERMS)
    signature_count = count_terms(text_lower, SIGNATURE_TERMS)
    title_count = count_terms(text_lower, TITLE_TERMS)
    country_count = count_terms(text_lower, COUNTRY_TERMS)
    organization_count = count_terms(text_lower, ORG_TERMS)

    starts_with_greeting = int(bool(re.match(r"^\s*(dear|hello|hi|greetings|good day)\b", text_lower)))
    ends_with_signature = int(
        bool(re.search(r"\b(regards|best regards|yours faithfully|sincerely|thank you|thanks)\b\s*[,!.]?\s*$", text_lower))
    )

    has_urgent_action_pattern = int(
        contains_any(text_lower, URGENCY_TERMS) and contains_any(text_lower, ACTION_TERMS)
    )
    has_confidentiality_pattern = int(contains_any(text_lower, SECRECY_TERMS))
    has_large_money_pattern = int(
        bool(re.search(r"[$€£]\s?\d{4,}|\b\d+(?:[,.]\d+)*\s?(?:million|billion)\b", text_lower))
        or "money_token" in text_lower
        or contains_any(text_lower, ["million", "billion", "inheritance", "lottery", "fortune"])
    )
    has_bank_details_request_pattern = int(
        contains_any(text_lower, BANK_TERMS)
        and contains_any(text_lower, ["send", "provide", "confirm", "fill", "reply"])
    )
    has_personal_info_request_pattern = int(
        contains_any(text_lower, PERSONAL_INFO_TERMS)
        and contains_any(text_lower, ["send", "provide", "confirm", "fill", "reply"])
    )
    has_advance_fee_pattern = int(
        contains_any(text_lower, FEE_TERMS)
        and contains_any(text_lower, ["payment", "pay", "send", "transfer", "clearance"])
    )
    body_is_empty = int(len(stripped) == 0)

    return {
        "char_count": float(char_count),
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "paragraph_count": float(paragraph_count),
        "line_count": float(line_count),
        "avg_word_length": round(float(avg_word_length), 6),
        "avg_sentence_length": round(float(avg_sentence_length), 6),
        "uppercase_ratio": round(float(uppercase_ratio), 6),
        "digit_ratio": round(float(digit_ratio), 6),
        "punctuation_count": float(punctuation_count),
        "exclamation_count": float(exclamation_count),
        "question_count": float(question_count),
        "multiple_exclamation_present": float(multiple_exclamation_present),
        "all_caps_word_count": float(all_caps_word_count),
        "url_count": float(url_count),
        "email_count": float(email_count),
        "phone_count": float(phone_count),
        "money_expression_count": float(money_expression_count),
        "anonymized_entity_token_count": float(anonymized_entity_token_count),
        "urgency_count": float(urgency_count),
        "secrecy_count": float(secrecy_count),
        "financial_count": float(financial_count),
        "bank_count": float(bank_count),
        "personal_info_count": float(personal_info_count),
        "action_request_count": float(action_request_count),
        "fee_count": float(fee_count),
        "greeting_count": float(greeting_count),
        "signature_count": float(signature_count),
        "title_count": float(title_count),
        "country_count": float(country_count),
        "organization_count": float(organization_count),
        "starts_with_greeting": float(starts_with_greeting),
        "ends_with_signature": float(ends_with_signature),
        "has_urgent_action_pattern": float(has_urgent_action_pattern),
        "has_confidentiality_pattern": float(has_confidentiality_pattern),
        "has_large_money_pattern": float(has_large_money_pattern),
        "has_bank_details_request_pattern": float(has_bank_details_request_pattern),
        "has_personal_info_request_pattern": float(has_personal_info_request_pattern),
        "has_advance_fee_pattern": float(has_advance_fee_pattern),
        "body_is_empty": float(body_is_empty),
    }


def detect_delimiter(sample: str) -> str:
    """Detect CSV delimiter. Falls back to comma."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except csv.Error:
        return ","


def read_csv_rows(path: Path, encoding: Optional[str] = None, delimiter: str = "auto") -> Tuple[List[Dict[str, str]], List[str]]:
    """Read a CSV file as a list of dictionaries, with encoding and delimiter fallback."""
    encodings: Sequence[str]
    if encoding:
        encodings = [encoding]
    else:
        encodings = ["utf-8-sig", "utf-8", "cp1253", "latin-1"]

    last_error: Optional[Exception] = None
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as file_obj:
                sample = file_obj.read(4096)
                file_obj.seek(0)
                selected_delimiter = detect_delimiter(sample) if delimiter == "auto" else delimiter
                reader = csv.DictReader(file_obj, delimiter=selected_delimiter)
                if not reader.fieldnames:
                    raise ValueError("CSV has no header row.")
                rows = [dict(row) for row in reader]
                return rows, list(reader.fieldnames)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except Exception as exc:  # noqa: BLE001 - keep exact file error for user-friendly CLI output
            last_error = exc
            break

    raise ValueError(f"Could not read {path.name}: {last_error}")


def find_column(fieldnames: Sequence[str], requested: Optional[str], candidates: Sequence[str], purpose: str) -> str:
    """Find a column by requested name or by common candidate names."""
    normalized_to_original = {clean_header_name(name): name for name in fieldnames}

    if requested:
        requested_key = clean_header_name(requested)
        if requested_key in normalized_to_original:
            return normalized_to_original[requested_key]
        raise ValueError(
            f"Requested {purpose} column '{requested}' was not found. Available columns: {list(fieldnames)}"
        )

    for candidate in candidates:
        candidate_key = clean_header_name(candidate)
        if candidate_key in normalized_to_original:
            return normalized_to_original[candidate_key]

    raise ValueError(
        f"Could not auto-detect {purpose} column. Available columns: {list(fieldnames)}. "
        f"Use --{purpose}-col to specify it manually."
    )


def get_csv_files(input_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    return files


def build_output_row(
    *,
    source_file: str,
    source_row: int,
    body: str,
    original_flag: str,
    flag_binary: int,
    drop_body: bool,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "source_file": source_file,
        "source_row": source_row,
        "flag": original_flag,
        "flag_binary": flag_binary,
        "category": "spam" if flag_binary == 1 else "legitimate",
    }

    if not drop_body:
        row["body"] = body

    row.update(extract_email_features(body))
    return row


def process_csv_files(
    *,
    input_dir: Path,
    output_path: Path,
    body_col: Optional[str],
    flag_col: Optional[str],
    recursive: bool,
    drop_body: bool,
    strict_labels: bool,
    encoding: Optional[str],
    delimiter: str,
) -> Tuple[int, int, Dict[str, int]]:
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input folder not found: {input_dir}. Create a folder named 'datasets' and put your CSV files inside it."
        )
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")

    csv_files = get_csv_files(input_dir, recursive=recursive)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    output_rows: List[Dict[str, Any]] = []
    skipped_rows = 0
    category_counts = {"spam": 0, "legitimate": 0}

    for csv_file in csv_files:
        rows, fieldnames = read_csv_rows(csv_file, encoding=encoding, delimiter=delimiter)
        detected_body_col = find_column(fieldnames, body_col, BODY_COLUMN_CANDIDATES, "body")
        detected_flag_col = find_column(fieldnames, flag_col, FLAG_COLUMN_CANDIDATES, "flag")

        print(
            f"Reading {csv_file.name}: rows={len(rows)}, "
            f"body_col='{detected_body_col}', flag_col='{detected_flag_col}'"
        )

        for index, input_row in enumerate(rows, start=2):  # line 1 is the header
            body = input_row.get(detected_body_col, "") or ""
            original_flag = input_row.get(detected_flag_col, "") or ""
            normalized = normalize_flag(original_flag)

            if normalized is None:
                message = (
                    f"Invalid flag value in file '{csv_file.name}', row {index}: {original_flag!r}. "
                    "Accepted spam examples: 1, true, spam, scam. "
                    "Accepted legitimate examples: 0, false, ham, legitimate."
                )
                if strict_labels:
                    raise ValueError(message)
                print(f"WARNING: {message} Row skipped.", file=sys.stderr)
                skipped_rows += 1
                continue

            output_row = build_output_row(
                source_file=csv_file.name,
                source_row=index,
                body=str(body),
                original_flag=str(original_flag),
                flag_binary=normalized,
                drop_body=drop_body,
            )
            output_rows.append(output_row)
            category_counts["spam" if normalized == 1 else "legitimate"] += 1

    if not output_rows:
        raise ValueError("No valid labelled rows were found. Output CSV was not created.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_columns = [col for col in OUTPUT_BASE_COLUMNS if not (drop_body and col == "body")]
    output_columns = base_columns + FEATURE_COLUMNS

    with output_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=output_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    return len(output_rows), skipped_rows, category_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically preprocess all labelled email CSV files from datasets/ into training_dataset.csv."
    )
    parser.add_argument(
        "--input-dir",
        default="datasets",
        help="Folder containing input CSV files. Default: datasets",
    )
    parser.add_argument(
        "--output",
        default="training_dataset.csv",
        help="Output CSV file. Default: training_dataset.csv",
    )
    parser.add_argument(
        "--body-col",
        default=None,
        help="Email body column name. Default: auto-detect, usually 'body'.",
    )
    parser.add_argument(
        "--flag-col",
        default=None,
        help="Spam label column name. Default: auto-detect, usually 'flag'.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also read CSV files inside subfolders of datasets/.",
    )
    parser.add_argument(
        "--drop-body",
        action="store_true",
        help="Do not include the body text in training_dataset.csv.",
    )
    parser.add_argument(
        "--strict-labels",
        action="store_true",
        help="Stop with an error when a row has an unknown flag value. Default: skip invalid rows with a warning.",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Input CSV encoding. Default: auto-try utf-8-sig, utf-8, cp1253, latin-1.",
    )
    parser.add_argument(
        "--delimiter",
        default="auto",
        help="CSV delimiter. Default: auto. Examples: ',', ';', tab, '|'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    try:
        total_rows, skipped_rows, category_counts = process_csv_files(
            input_dir=input_dir,
            output_path=output_path,
            body_col=args.body_col,
            flag_col=args.flag_col,
            recursive=args.recursive,
            drop_body=args.drop_body,
            strict_labels=args.strict_labels,
            encoding=args.encoding,
            delimiter=args.delimiter,
        )
    except Exception as exc:  # noqa: BLE001 - useful for a command-line script
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nDone.")
    print(f"Output CSV: {output_path}")
    print(f"Valid rows written: {total_rows}")
    print(f"Skipped rows: {skipped_rows}")
    print(f"Category counts: {category_counts}")
    print(f"Feature columns added: {len(FEATURE_COLUMNS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
