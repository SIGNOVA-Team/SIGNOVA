# ArabSign Dataset — Data Folder Guide

مجلد داتا سيت **ArabSign** مخصص لأبحاث لغة الإشارة العربية المستمرة (Continuous Arabic Sign Language - ArSL).

---

## 📂 محتويات المجلد

### 1. `metadata/`
- **`arabsign_canonical_metadata.csv`**:
  - يحتوي على **9,335** عينة محققة وفق مخطط SIGNOVA المعياري.
  - الحقول تشمل: `sample_id`, `sign_language (ArSL)`, `original_label`, `gloss_label`, `signer_id (01-06)`, `duration_sec`, `fps (30.0)`, `qc_flags (DIALECT_NOT_ESL)`.
- **`arabsign_sentences_mapping.csv`**:
  - يحتوي على الـ **50 جملة** المعتمدة مع ترجمتها بالإنجليزية والربط بين:
    - **ArSL Gloss:** ترتيب الإشارات الفعلي (مثل: `الحمد الله`، `الصم تمني اشارة-هم جميع السامع`).
    - **Spoken Arabic Standard:** الصياغة العربية الفصحى المقابلة (مثل: `حمد الله`، `تمني الصم اشارتهم لجميع السامعين`).

### 2. `splits/`
- **`arabsign_split_manifest_signer_independent.csv` (الموصى به - Recommended):**
  - **Train:** Signers 01, 02, 03, 04 (6,323 عينة $\approx 67.7\%$)
  - **Val:** Signer 05 (1,500 عينة $\approx 16.1\%$)
  - **Test:** Signer 06 (1,512 عينة $\approx 16.2\%$)
  - **ميزة حاسمة:** يضمن **0% تداخل** في الأشخاص الموقعين (Signers) لاختبار قدرة النماذج على التعميم على أشخاص جدد لم يسبق رؤيتهم.
- **`arabsign_split_manifest_signer_dependent.csv`:**
  - تقسيم الـ 6-fold الخاص بالمؤلف الأصلي لمقارنة النتائج البحثية بالأوراق العلمية المنشورة.

### 3. `raw/`
- **`groundTruth.txt`**: قائمة الجمل الـ 50 الأصلية كما وردت من الباحث.
- **`author_splits/`**: ملفات CSV الأصلية للتدريب والاختبار لكل Signer من `01_train.csv` حتى `06_test.csv`.

---

## ⚠️ تنبيه لغوي هام
- داتا ArabSign تعبر عن لغة الإشارة المعتمدة في السعودية / الخليج العربي، وليست لغة الإشارة المصرية (ESL).
- راجع وثيقة القرار المعماري: [`docs/decisions/ADR_001_p2_datasets_evaluation.md`](../../docs/decisions/ADR_001_p2_datasets_evaluation.md).
