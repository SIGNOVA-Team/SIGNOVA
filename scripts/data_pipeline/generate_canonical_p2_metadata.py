"""
SIGNOVA - P2 Metadata Pipeline Generator
Parses and standardizes metadata for:
  - P2: How2Sign (Continuous ASL)
  - P2: ArabSign (Continuous ArSL)
Generates:
  - Canonical metadata CSVs
  - Split manifests (both standard and signer-independent)
  - ArabSign sentence mappings
"""

import os
import sys
import csv
import re
import glob

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")

CANONICAL_COLUMNS = [
    "sample_id",
    "source_dataset",
    "source_sample_id",
    "sign_language",
    "original_label",
    "gloss_label",
    "signer_id",
    "session_id",
    "relative_video_path",
    "fps",
    "duration_sec",
    "width",
    "height",
    "split",
    "split_version",
    "license_id",
    "qc_status",
    "qc_flags",
    "checksum"
]


def process_how2sign(source_dir, output_meta_path, output_split_path):
    print("Processing How2Sign...")
    splits = {
        "train": os.path.join(source_dir, "scratch_how2sign_train.csv"),
        "val": os.path.join(source_dir, "scratch_how2sign_val.csv"),
        "test": os.path.join(source_dir, "scratch_how2sign_test.csv")
    }

    canonical_rows = []
    split_rows = []

    for split_name, filepath in splits.items():
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Skipping {split_name}.")
            continue

        with open(filepath, mode="r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for idx, r in enumerate(reader):
                video_id = r.get("VIDEO_ID", "").strip()
                video_name = r.get("VIDEO_NAME", "").strip()
                sentence_id = r.get("SENTENCE_ID", "").strip()
                sentence_name = r.get("SENTENCE_NAME", "").strip()
                start_val = r.get("START", "").strip()
                end_val = r.get("END", "").strip()
                sentence = r.get("SENTENCE", "").strip()

                # Extract signer ID from video name, e.g. -d5dN54tH2E-1-rgb_front -> 1
                signer_match = re.search(r"-(\d+)-rgb", video_name)
                signer_id = signer_match.group(1) if signer_match else "unknown"

                # Calculate duration
                try:
                    duration = round(float(end_val) - float(start_val), 3)
                except (ValueError, TypeError):
                    duration = -1.0

                sample_id = f"H2S_{sentence_id}_{signer_id}"
                rel_path = f"videos/how2sign/{split_name}/rgb_front/{sentence_name}.mp4"

                # Check QC flags: In official How2Sign splits, signers 1, 2, 3, 5, 8 overlap across train/val/test
                qc_status = "PASS"
                qc_flags = "NONE"
                if signer_id in ["1", "2", "3", "5", "8"]:
                    qc_flags = "SIGNER_OVERLAP_ACROSS_SPLITS"

                canonical_rows.append({
                    "sample_id": sample_id,
                    "source_dataset": "How2Sign",
                    "source_sample_id": sentence_name,
                    "sign_language": "ASL",
                    "original_label": sentence,
                    "gloss_label": sentence,  # Aligned English text
                    "signer_id": signer_id,
                    "session_id": video_id,
                    "relative_video_path": rel_path,
                    "fps": 25.0,
                    "duration_sec": duration,
                    "width": 1920,
                    "height": 1080,
                    "split": split_name,
                    "split_version": "v1.0-official",
                    "license_id": "CC-BY-NC-4.0",
                    "qc_status": qc_status,
                    "qc_flags": qc_flags,
                    "checksum": "NOT_CALCULATED_REMOTE"
                })

                split_rows.append({
                    "sample_id": sample_id,
                    "source_sample_id": sentence_name,
                    "signer_id": signer_id,
                    "session_id": video_id,
                    "split": split_name,
                    "split_version": "v1.0-official",
                    "rationale": "Official How2Sign topic/video partition (inherits How2 speech splits; signer overlap present)"
                })

    # Write canonical metadata
    with open(output_meta_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        writer.writerows(canonical_rows)
    print(f"Wrote {len(canonical_rows)} How2Sign canonical rows to {output_meta_path}")

    # Write split manifest
    with open(output_split_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "source_sample_id", "signer_id", "session_id", "split", "split_version", "rationale"])
        writer.writeheader()
        writer.writerows(split_rows)
    print(f"Wrote {len(split_rows)} How2Sign split rows to {output_split_path}")


def process_arabsign(extracted_dir, output_meta_path, output_split_dep_path, output_split_indep_path, output_sentences_path):
    print("Processing ArabSign...")
    data_dir = os.path.join(extracted_dir, "ArabSign-EncoderDec", "EncoderDec", "data")
    gt_path = os.path.join(data_dir, "groundTruth.txt")

    # Load 50 ground truth sentences
    gt_sentences = {}
    if os.path.exists(gt_path):
        with open(gt_path, mode="r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, 1):
                clean_line = line.strip()
                if clean_line:
                    gt_sentences[idx] = clean_line

    # English approximate translations & domain category mapping for the 50 sentences
    sentence_info = {
        1: ("In the name of Allah", "Religion", "اسم الله"),
        2: ("Praise be to Allah", "Religion", "الحمد الله"),
        3: ("All Arab Deaf and Hearing people", "Deaf Culture", "جميع الصم العرب  السامع"),
        4: ("Peace be upon you, and Allah's mercy and blessings", "Greeting", "السلام عليكم رحمة الله بركة"),
        5: ("Today I present to you another program", "Presentation", "اليوم اقدم انتم برنامج اخر"),
        6: ("Subject of studying Arabic Sign Language", "Education", "موضوع دراسة لغة الاشارة العربية"),
        7: ("Today's words are varied in religion", "Religion", "كلمات اليوم متفرقة في الدين"),
        8: ("Also ordinary words", "General", "ايضا كلمات عادية"),
        9: ("No partners with Allah", "Religion", "لا شرك الله"),
        10: ("Allah is the Greatest", "Religion", "الله اكبر"),
        11: ("No partners with Allah, the One", "Religion", "لا شرك الله احد"),
        12: ("Worship is for Allah alone", "Religion", "العبادة الله واحد"),
        13: ("Allah is Generous", "Religion", "الله كريم"),
        14: ("Allah is the Provider", "Religion", "الله رزق"),
        15: ("Also different words", "General", "ايضا كلمات مختلف"),
        16: ("All of us are poor before Allah", "Religion", "كلنا فقير الله"),
        17: ("Allah is Self-Sufficient / Rich", "Religion", "الله غني"),
        18: ("Allah helps all people", "Religion", "الله يساعد جميع الناس"),
        19: ("Life was created by Allah", "Religion", "الحياة خلق الله"),
        20: ("To whom? All people", "General", "لمن جميع الناس"),
        21: ("Allah created people for worship", "Religion", "الله خلق الناس عبادة"),
        22: ("Thank you", "Social", "شكرا انتم"),
        23: ("You are always welcome", "Social", "ترحيب انتم دائما"),
        24: ("Also wish his sign is cooperation to you", "Deaf Culture", "ايضا تمني اشارة-هو التعاون لكم"),
        25: ("Teaching sign language with gestures/mimics", "Education", "تعليم لغة الاشارة مع الامائيات"),
        26: ("I love learning sign language", "Education", "انا حب تعليم لغة الاشارة"),
        27: ("I travel", "Daily Life", "انا سفر"),
        28: ("Where is Saudi Arabia?", "Geography", "اين السعودية"),
        29: ("Tomorrow there might be an exam", "Education", "غدا احتمال في امتحان"),
        30: ("The body interacts with the word", "Sign Linguistics", "الجسم تفاعل مع الكلمة"),
        31: ("Deaf wish their sign is for all hearing", "Deaf Culture", "الصم تمني اشارة-هم جميع السامع"),
        32: ("Interact with the deaf, communicate", "Deaf Culture", "تفاعل مع الصم تواصل"),
        33: ("Body interacts with...", "Sign Linguistics", "تفاعل الجسم مع"),
        34: ("Here we conclude / that is enough", "Social", "هنا نكتفي"),
        35: ("May Allah bring meeting with you", "Social / Religious", "الله لقاء انتم"),
        36: ("May Allah accept the work/deeds", "Religion", "الله قبول عمل"),
        37: ("Wish from Allah benefit", "Religion", "تمني الله فائدة"),
        38: ("We and you all", "Social", "نحن انتم جميع"),
        39: ("First word", "General", "كلمة اول"),
        40: ("I seek refuge in Allah", "Religion", "الله اعوذ"),
        41: ("Medical advice / health tips", "Health", "نصائح طبية"),
        42: ("Subject of Coronavirus", "Health", "موضوع فايروس كورونا"),
        43: ("Part two", "General", "الجزء الثاني"),
        44: ("Day of Resurrection", "Religion", "يوم القيامة"),
        45: ("The five prayers", "Religion", "الصلة الخمس"),
        46: ("Friday sermon", "Religion", "خطبة الجمعة"),
        47: ("Tomorrow meeting, in the name of Allah", "Social / Religious", "لقاء غدا اسم الله"),
        48: ("Continuance to you / carry on", "Social", "استمرار انتم"),
        49: ("Allah knows best", "Religion", "الله اعلم"),
        50: ("Acquisition of sign language", "Education", "اكتساب لغة الاشارة")
    }

    # Write arabsign_sentences_mapping.csv
    sentence_rows = []
    for s_id in range(1, 51):
        gt_text = gt_sentences.get(s_id, "")
        info = sentence_info.get(s_id, ("Unknown", "General", gt_text))
        sentence_rows.append({
            "sentence_id": s_id,
            "arsl_gloss_structure": gt_text,
            "spoken_arabic_standard": info[2],
            "english_translation": info[0],
            "domain_category": info[1],
            "sign_count": len(gt_text.split()),
            "source_origin": "KFUPM / Hamzah Luqman (2023) - Saudi/Pan-Arab ArSL"
        })

    with open(output_sentences_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence_id", "arsl_gloss_structure", "spoken_arabic_standard", "english_translation", "domain_category", "sign_count", "source_origin"])
        writer.writeheader()
        writer.writerows(sentence_rows)
    print(f"Wrote 50 ArabSign sentence mappings to {output_sentences_path}")

    # Process all sample CSVs
    all_files = sorted(glob.glob(os.path.join(data_dir, "*_train.csv")) + glob.glob(os.path.join(data_dir, "*_test.csv")))
    
    unique_samples = {}
    signer_dep_splits = []

    for fpath in all_files:
        fname = os.path.basename(fpath)
        fold_id = fname.split("_")[0]  # '01' to '06'
        sub_split = fname.split("_")[1].replace(".csv", "")  # 'train' or 'test'

        with open(fpath, mode="r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 7:
                    continue
                orig_split = row[0].strip()
                sentence_num = int(row[1].strip())
                orig_path = row[2].strip()
                max_len = row[3].strip()
                class_id = row[4].strip()
                arsl_gloss = row[5].strip()
                arabic_text = row[6].strip()

                # Extract signer from path: e.g. /home/eye/ArSL-Continuous/80/color/01/train/0001/01_0001_(10_03_21_20_37_17)_c
                signer_match = re.search(r"/color/(\d+)/", orig_path)
                signer_id = signer_match.group(1) if signer_match else fold_id

                sample_basename = os.path.basename(orig_path)
                sample_id = f"ARAB_{signer_id}_{sentence_num:02d}_{sample_basename}"

                # Store for canonical sample metadata (deduplicated by sample_id)
                if sample_id not in unique_samples:
                    unique_samples[sample_id] = {
                        "sample_id": sample_id,
                        "source_dataset": "ArabSign",
                        "source_sample_id": sample_basename,
                        "sign_language": "ArSL",
                        "original_label": arabic_text,
                        "gloss_label": arsl_gloss,
                        "signer_id": signer_id,
                        "session_id": f"signer_{signer_id}_sess",
                        "relative_video_path": f"videos/arabsign/color/{signer_id}/{sample_basename}.mp4",
                        "fps": 30.0,
                        "duration_sec": round(float(max_len) / 30.0, 3) if max_len.isdigit() else 2.67,
                        "width": 1920,
                        "height": 1080,
                        "split": "all_pool",  # Specific split assigned in manifests
                        "split_version": "v1.0",
                        "license_id": "Non-Commercial-Academic-KFUPM",
                        "qc_status": "PASS",
                        "qc_flags": "DIALECT_NOT_ESL",  # Critical reminder: ArSL is NOT ESL
                        "checksum": "NOT_CALCULATED_REMOTE"
                    }

                # Record signer-dependent fold split
                signer_dep_splits.append({
                    "sample_id": sample_id,
                    "fold_signer": fold_id,
                    "split": sub_split,
                    "split_version": f"v1.0-fold{fold_id}",
                    "rationale": f"Author signer-dependent split for signer {fold_id} (train={orig_split})"
                })

    # Build canonical rows
    canonical_rows = list(unique_samples.values())
    canonical_rows.sort(key=lambda x: x["sample_id"])

    with open(output_meta_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        writer.writerows(canonical_rows)
    print(f"Wrote {len(canonical_rows)} ArabSign canonical samples to {output_meta_path}")

    # Write signer-dependent split manifest
    with open(output_split_dep_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "fold_signer", "split", "split_version", "rationale"])
        writer.writeheader()
        writer.writerows(signer_dep_splits)
    print(f"Wrote {len(signer_dep_splits)} ArabSign signer-dependent split rows to {output_split_dep_path}")

    # Generate SIGNOVA Group-Aware Signer-Independent Split:
    # 6 signers: Signers 01, 02, 03, 04 -> TRAIN (~67.7%)
    #            Signer 05 -> VAL (~16.1%)
    #            Signer 06 -> TEST (~16.2%)
    signer_indep_splits = []
    for s in canonical_rows:
        sid = s["signer_id"]
        if sid in ["01", "02", "03", "04"]:
            part = "train"
        elif sid == "05":
            part = "val"
        elif sid == "06":
            part = "test"
        else:
            part = "train"

        signer_indep_splits.append({
            "sample_id": s["sample_id"],
            "source_sample_id": s["source_sample_id"],
            "signer_id": sid,
            "split": part,
            "split_version": "v1.0-signer-independent",
            "rationale": f"Signer-independent partition (Signers 01-04 train, 05 val, 06 test) preventing signer leakage"
        })

    with open(output_split_indep_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "source_sample_id", "signer_id", "split", "split_version", "rationale"])
        writer.writeheader()
        writer.writerows(signer_indep_splits)
    print(f"Wrote {len(signer_indep_splits)} ArabSign signer-independent split rows to {output_split_indep_path}")


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    workspace_root = os.path.abspath(os.path.join(root, ".."))

    # How2Sign
    h2s_meta = os.path.join(METADATA_DIR, "how2sign_canonical_metadata.csv")
    h2s_split = os.path.join(SPLITS_DIR, "how2sign_split_manifest.csv")
    process_how2sign(workspace_root, h2s_meta, h2s_split)

    # ArabSign
    arab_extracted = os.path.join(workspace_root, "scratch_arabsign_extracted")
    arab_meta = os.path.join(METADATA_DIR, "arabsign_canonical_metadata.csv")
    arab_split_dep = os.path.join(SPLITS_DIR, "arabsign_split_manifest_signer_dependent.csv")
    arab_split_indep = os.path.join(SPLITS_DIR, "arabsign_split_manifest_signer_independent.csv")
    arab_sentences = os.path.join(METADATA_DIR, "arabsign_sentences_mapping.csv")
    process_arabsign(arab_extracted, arab_meta, arab_split_dep, arab_split_indep, arab_sentences)

    print("Canonical metadata processing completed successfully.")


if __name__ == "__main__":
    main()
