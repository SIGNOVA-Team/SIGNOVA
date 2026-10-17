# How2Sign Dataset — Data Folder Guide

مجلد داتا سيت **How2Sign** مخصص لأبحاث لغة الإشارة الأمريكية المستمرة (Continuous American Sign Language - ASL).

---

## 📂 محتويات المجلد

### 1. `metadata/`
- **`how2sign_canonical_metadata.csv`**:
  - يحتوي على **35,263** جملة كاملة ومحققة وفق مخطط SIGNOVA المعياري.
  - الحقول تشمل: `sample_id`, `sign_language (ASL)`, `original_label`, `signer_id (1-11)`, `duration_sec`, `fps (25.0)`, `qc_flags (SIGNER_OVERLAP_ACROSS_SPLITS)`.

### 2. `splits/`
- **`how2sign_split_manifest.csv`**:
  - التقسيم الرسمي المعتمد في مجتمع أبحاث How2Sign:
    - **Train:** 31,165 جملة (88.4%)
    - **Val:** 1,741 جملة (4.9%)
    - **Test:** 2,357 جملة (6.7%)
  - **ملاحظة فحص الجودة (QC-P2-001):** هذا التقسيم مبني على مقاطع الفيديو والمواضيع (Topics)، وتوجد 5 إشارات (Signers 1, 2, 3, 5, 8) مشتركة بين مجموعات التدريب والاختبار.

### 3. `raw/`
الملفات الأصلية المنزلة مباشرة من جوجل درايف الخاص بالمشروع الرسمي:
- **`how2sign_train.csv` / `how2sign_val.csv` / `how2sign_test.csv`**: النصوص الأصلية للجمل وبداية ونهاية كل جملة بالثواني.
- **`how2sign_realigned_train.csv` / `how2sign_realigned_val.csv` / `how2sign_realigned_test.csv`**: النسخة المحسنة المعاد محاذاتها زمنياً بدقة أعلى.

---

## 🔗 روابط ووثائق مساعدة
- تفاصيل التراخيص والأحجام: [`docs/dataset_card_how2sign.md`](../../docs/dataset_card_how2sign.md).
- سجل التنزيلات الكامل لملفات الفيديو و2D Keypoints: [`data/reports/download_manifest.csv`](../reports/download_manifest.csv).
