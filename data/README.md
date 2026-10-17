# SIGNOVA Data Directory Structure

هذا المجلد يحتوي على جميع البيانات المجهزة والمحققة لمشروع **SIGNOVA**، منظمة بشكل معياري ومنفصل لكل داتا سيت لمنع التداخل وتسهيل العمل والتدريب.

---

## 📁 هيكل المجلدات المنظم (Organized Directory Tree)

```text
data/
│
├── arabsign/                                    # 🇸🇦 داتا سيت ArabSign (لغة الإشارة العربية المستمرة)
│   ├── README.md                                # دليل مفصل لداتا ArabSign والتقسيمات
│   ├── metadata/
│   │   ├── arabsign_canonical_metadata.csv      # الميتاداتا الموحدة لجميع العينات (9,335 عينة)
│   │   └── arabsign_sentences_mapping.csv       # الـ 50 جملة مع الربط بين الإشارة والنص العربي
│   ├── splits/
│   │   ├── arabsign_split_manifest_signer_independent.csv  # التقسيم المعتمد (بدون أي تسريب إشارات)
│   │   └── arabsign_split_manifest_signer_dependent.csv    # تقسيم الـ 6-fold الخاص بالمؤلف
│   └── raw/                                     # البيانات الخام المستخرجة
│       ├── groundTruth.txt                      # نصوص الجمل الأصلية الـ 50
│       └── author_splits/                       # تقسيمات المؤلف الأصلية (01 إلى 06 train/test)
│
├── how2sign/                                    # 🇺🇸 داتا سيت How2Sign (لغة الإشارة الأمريكية المستمرة)
│   ├── README.md                                # دليل مفصل لداتا How2Sign
│   ├── metadata/
│   │   └── how2sign_canonical_metadata.csv      # الميتاداتا الموحدة لـ 35,263 جملة
│   ├── splits/
│   │   └── how2sign_split_manifest.csv          # التقسيم الرسمي (Train / Val / Test)
│   └── raw/                                     # النصوص والترجمات الخام المنزلة رسمياً
│       ├── how2sign_train.csv
│       ├── how2sign_val.csv
│       ├── how2sign_test.csv
│       ├── how2sign_realigned_train.csv
│       ├── how2sign_realigned_val.csv
│       └── how2sign_realigned_test.csv
│
└── reports/                                     # 📊 التقارير المشتركة، السجلات، وضمان الجودة
    ├── README.md                                # دليل التقارير والمصادر
    ├── dataset_sources.csv                      # سجل المصادر الرسمية والتراخيص (P0, P1, P2)
    ├── download_manifest.csv                    # روابط وأحجام وأذونات جميع الوسائط والوسائط المتعددة
    ├── qc_flags_p2.csv                          # جدول ملاحظات فحص الجودة (Quality Flags)
    └── qc_report_p2.md                          # التقرير الفني واللغوي المفصل
```

---

## 📌 القواعد والمعايير الأساسية (Key Guidelines)

1. **فصل لغة الإشارة العربية عن المصرية:**
   - داتا **ArabSign** تم جمعها في جامعة الملك فهد للبترول والمعادن (KFUPM) بالسعودية وتعبر عن لغة الإشارة العربية (ArSL).
   - **قاعدة صارمة:** لا يجوز إطلاق اسم "لغة الإشارة المصرية (ESL)" عليها أو دمجها في نماذج ESL دون مراجعة لغوية متخصصة.
2. **منع تسريب الإشارات (No Signer Leakage):**
   - في ArabSign، تم اعتماد `arabsign_split_manifest_signer_independent.csv`:
     - **Train:** Signers 01, 02, 03, 04 (6,323 عينة)
     - **Val:** Signer 05 (1,500 عينة)
     - **Test:** Signer 06 (1,512 عينة)
     - نسبة تداخل الإشارات بين المجموعات: **0% (صفر تسريب)**.
3. **الوسائط الضخمة (Large Media & Videos):**
   - ملفات الفيديو الكبيرة (`*.mp4`, `*.zip`, `*.tar.gz`) لا تُرفع على Git ومدرجة في `.gitignore`.
   - ملفات الميتاداتا والنصوص (`*.csv`, `*.txt`) محفوظة وتكفي لجميع تجارب النماذج المعتمدة على الميزات أو الـ Landmarks.
