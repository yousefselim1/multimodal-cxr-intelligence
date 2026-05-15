"""
QA Dataset Generator for Chest X-Ray Reports.
Creates question-answer pairs from radiology report text using rule-based templates.
"""

import re
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.config import QA_DATASET_PATH
except ImportError:
    QA_DATASET_PATH = Path("./data/qa_dataset.csv")

# Regex patterns to detect medical findings
FINDING_PATTERNS = {
    "cardiomegaly": {
        "positive": [r"cardio\w*megaly", r"heart\s+(is\s+)?(enlarged|enlargement)", r"cardiac\s+silhouette\s+(is\s+)?(enlarged|prominent)"],
        "negative": [r"no\s+cardio\w*megaly", r"heart\s+(size\s+)?(is\s+)?(normal|within\s+normal)"],
    },
    "pleural_effusion": {
        "positive": [r"pleural\s+effusion", r"(small|moderate|large)\s+effusion", r"costophrenic\s+angle\s+(is\s+)?(blunted|obscured)"],
        "negative": [r"no\s+pleural\s+effusion", r"no\s+(significant\s+)?effusion", r"costophrenic\s+angles?\s+(are\s+)?(sharp|clear)"],
    },
    "pneumothorax": {
        "positive": [r"pneumothorax"],
        "negative": [r"no\s+pneumothorax", r"no\s+evidence\s+of\s+pneumothorax"],
    },
    "atelectasis": {
        "positive": [r"atelecta\w+", r"(bibasilar|basilar|subsegmental)\s+atelecta\w+"],
        "negative": [r"no\s+atelecta\w+"],
    },
    "consolidation": {
        "positive": [r"consolidat\w+", r"pneumonia", r"airspace\s+(disease|opacity)"],
        "negative": [r"no\s+(focal\s+)?consolidat\w+", r"no\s+(evidence\s+of\s+)?pneumonia"],
    },
    "edema": {
        "positive": [r"(pulmonary\s+)?edema", r"vascular\s+(congestion|engorgement)", r"cephalization"],
        "negative": [r"no\s+(pulmonary\s+)?edema", r"no\s+vascular\s+congestion"],
    },
    "lung_opacity": {
        "positive": [r"opaci\w+", r"infiltrat\w+", r"(hazy|haziness)"],
        "negative": [r"lungs?\s+(are\s+)?(clear|free)", r"no\s+(focal\s+)?opaci"],
    },
    "support_devices": {
        "positive": [r"(endotracheal|ET|NG)\s+tube", r"(central\s+venous|PICC)\s+(catheter|line)", r"pacemaker", r"sternotomy\s+wires?"],
        "negative": [r"no\s+(support\s+)?devices?", r"no\s+(lines?|tubes?)"],
    },
    "fracture": {
        "positive": [r"fractur\w+"],
        "negative": [r"no\s+(acute\s+)?(bony\s+)?fractur\w+", r"no\s+osseous\s+abnormality"],
    },
    "no_finding": {
        "positive": [r"no\s+acute\s+(cardiopulmonary\s+)?(process|abnormality|finding)", r"(normal|unremarkable)\s+(chest|study)", r"within\s+normal\s+limits"],
        "negative": [],
    },
}

SEVERITY_PATTERNS = {
    "mild": r"\bmild(ly)?\b", "moderate": r"\bmoderate(ly)?\b", "severe": r"\bsevere(ly)?\b",
    "small": r"\bsmall\b", "large": r"\blarge\b", "minimal": r"\bminimal\b",
    "marked": r"\bmarked\b", "extensive": r"\bextensive\b", "stable": r"\bstable\b|unchanged",
}

LOCATION_PATTERNS = {
    "right_lung": r"right\s+(lung|lobe|hemithorax)", "left_lung": r"left\s+(lung|lobe|hemithorax)",
    "bilateral": r"bilateral|both\s+lungs?", "upper_lobe": r"(upper|apical)\s+(lobe|zone)",
    "lower_lobe": r"(lower|basilar)\s+(lobe|zone)", "perihilar": r"perihilar|hilar",
}


def extract_sentence(text, char_position):
    """Extract the sentence containing the character at the given position."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_pos = 0
    for sentence in sentences:
        end_pos = current_pos + len(sentence)
        if current_pos <= char_position < end_pos:
            return sentence.strip()
        current_pos = end_pos + 1
    start = max(0, char_position - 50)
    end = min(len(text), char_position + 100)
    return text[start:end].strip()


def extract_findings(report_text):
    """Extract medical findings from a report using regex patterns."""
    if not report_text or not isinstance(report_text, str):
        return {}
    text_lower = report_text.lower()
    findings = {}
    for category, patterns in FINDING_PATTERNS.items():
        finding = {"present": None, "evidence": [], "severity": None, "location": None}
        for pattern in patterns.get("positive", []):
            match = re.search(pattern, text_lower)
            if match:
                finding["present"] = True
                finding["evidence"].append(extract_sentence(report_text, match.start()))
                break
        if finding["present"] is None:
            for pattern in patterns.get("negative", []):
                match = re.search(pattern, text_lower)
                if match:
                    finding["present"] = False
                    finding["evidence"].append(extract_sentence(report_text, match.start()))
                    break
        if finding["present"]:
            for sev_name, sev_pat in SEVERITY_PATTERNS.items():
                for ev in finding["evidence"]:
                    if re.search(sev_pat, ev.lower()):
                        finding["severity"] = sev_name
                        break
                if finding["severity"]: break
            for loc_name, loc_pat in LOCATION_PATTERNS.items():
                for ev in finding["evidence"]:
                    if re.search(loc_pat, ev.lower()):
                        finding["location"] = loc_name
                        break
                if finding["location"]: break
        if finding["present"] is not None:
            findings[category] = finding
    return findings


def generate_qa_pairs_for_report(report_text, image_path=None, sample_id=None):
    """Generate QA pairs from a single radiology report."""
    qa_pairs = []
    findings = extract_findings(report_text)
    if not findings:
        return qa_pairs

    def make_qa(question, answer, category, evidence, method="template_rule_based"):
        return {"sample_id": sample_id, "image_path": image_path, "report_text": report_text,
                "question": question, "answer": answer, "category": category,
                "evidence_sentence": evidence, "generation_method": method}

    # 1. Presence/Absence Questions
    for cat, f in findings.items():
        if cat == "no_finding": continue
        cat_display = cat.replace("_", " ")
        ev = f["evidence"][0] if f["evidence"] else ""
        if f["present"] is True:
            qa_pairs.append(make_qa(f"Is there evidence of {cat_display} in this chest X-ray?",
                f"Yes. The report indicates {cat_display}. {ev}", cat, ev))
        elif f["present"] is False:
            qa_pairs.append(make_qa(f"Is there {cat_display} present in this chest X-ray?",
                f"No. The report states there is no {cat_display}. {ev}", cat, ev))

    # 2. Severity Questions
    for cat, f in findings.items():
        if f["present"] and f["severity"]:
            cat_display = cat.replace("_", " ")
            ev = f["evidence"][0] if f["evidence"] else ""
            qa_pairs.append(make_qa(f"What is the severity of the {cat_display}?",
                f"The {cat_display} is described as {f['severity']}. {ev}", "severity", ev))

    # 3. Location Questions
    for cat, f in findings.items():
        if f["present"] and f["location"]:
            cat_display, loc_display = cat.replace("_", " "), f["location"].replace("_", " ")
            ev = f["evidence"][0] if f["evidence"] else ""
            qa_pairs.append(make_qa(f"Where is the {cat_display} located?",
                f"The {cat_display} is located in the {loc_display}. {ev}", "location", ev))

    # 4. Summary Question
    pos = [c.replace("_"," ") for c,f in findings.items() if f["present"] is True and c!="no_finding"]
    neg = [c.replace("_"," ") for c,f in findings.items() if f["present"] is False]
    if pos:
        ans = f"The main findings include: {', '.join(pos)}."
        if neg: ans += f" Notably absent: {', '.join(neg[:3])}."
    elif "no_finding" in findings and findings["no_finding"]["present"]:
        ans = "No significant acute findings are identified. The study appears normal."
    else:
        ans = "The report does not describe specific notable findings."
    qa_pairs.append(make_qa("What are the main findings in this chest X-ray?", ans, "summary", report_text[:200]))

    # 5. Device Question
    if "support_devices" in findings:
        df_ = findings["support_devices"]
        ev = df_["evidence"][0] if df_["evidence"] else ""
        if df_["present"]:
            qa_pairs.append(make_qa("Are there any support devices visible?", f"Yes. {ev}", "support_devices", ev))
        else:
            qa_pairs.append(make_qa("Are there any lines, tubes, or support devices present?",
                "No. The report does not mention any support devices.", "support_devices", ev))

    # 6. Normal Study Question
    if "no_finding" in findings and findings["no_finding"]["present"]:
        ev = findings["no_finding"]["evidence"][0] if findings["no_finding"]["evidence"] else ""
        qa_pairs.append(make_qa("Is this chest X-ray normal?",
            f"Yes. The report indicates a normal study. {ev}", "no_finding", ev))

    # 7. Comparison Question
    if len(pos) >= 2:
        qa_pairs.append(make_qa(f"Does this chest X-ray show both {pos[0]} and {pos[1]}?",
            f"Yes. The report identifies both {pos[0]} and {pos[1]}.", "comparison", report_text[:200]))

    return qa_pairs


def generate_qa_dataset(df, output_path=None):
    """Generate QA dataset from the full preprocessed DataFrame."""
    if output_path is None:
        output_path = QA_DATASET_PATH
    all_qa = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating QA"):
        pairs = generate_qa_pairs_for_report(
            row.get("report_text", ""), row.get("image_path"), row.get("sample_id", f"cxr_{idx:05d}"))
        all_qa.extend(pairs)
    qa_df = pd.DataFrame(all_qa)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    qa_df.to_csv(output_path, index=False)
    logger.info(f"Generated {len(qa_df)} QA pairs from {qa_df['sample_id'].nunique()} reports")
    logger.info(f"Category distribution:\n{qa_df['category'].value_counts()}")
    return qa_df


if __name__ == "__main__":
    test_reports = [
        "The heart is mildly enlarged. There is a small left pleural effusion. No pneumothorax.",
        "No acute cardiopulmonary process. Heart size is normal. Lungs are clear.",
        "ET tube in satisfactory position. Bilateral pulmonary edema with bilateral pleural effusions. Moderate cardiomegaly.",
        "Right lower lobe consolidation consistent with pneumonia. Small right pleural effusion. Heart size is normal.",
    ]
    for i, r in enumerate(test_reports):
        pairs = generate_qa_pairs_for_report(r, f"test_{i}.jpg", f"test_{i:03d}")
        print(f"\n--- Report {i} ({len(pairs)} QA pairs) ---")
        for p in pairs:
            print(f"  Q: {p['question']}\n  A: {p['answer']}\n  Cat: {p['category']}\n")
