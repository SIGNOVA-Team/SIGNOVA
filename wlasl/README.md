# WLASL (Word-Level American Sign Language) Module

هذا المجلد يحتوي على أدوات المعالجة، تنظيف الميتاداتا، وفحص صلاحية الروابط لـ **WLASL2000** (المصدر الثانوي المعتمد للغة الإشارة الأمريكية في مشروع SIGNOVA).

---

## 📁 هيكل المجلد المنظم (Directory Layout)

```text
wlasl/
├── README.md                            # هذا الدليل التوثيقي
├── docs/
│   └── wlasl_link_check_report.md       # تقرير فحص صلاحية الروابط الشامل
├── scripts/                             # سكربتات المعالجة والاستخراج
│   ├── check_links.py                   # أداة فحص صلاحية روابط الفيديوهات (HEAD / yt-dlp)
│   ├── clean_metadata.py                # تنظيف وتوحيد ميتاداتا WLASL_v0.3.json
│   └── extract_keypoints.py             # استخراج نقاط MediaPipe Holistic للروابط الصالحة
└── data/                                # البيانات والميتاداتا المحققة
    ├── label_map.csv                    # خريطة الفئات الـ 2000 (Gloss <-> Label ID)
    ├── summary.txt                      # ملخص أعداد الفئات والروابط
    ├── url_status.csv                   # حالة كل رابط فريد (ok / dead / unchecked)
    ├── wlasl2000_ok.csv                 # الفيديوهات الصالحة للاستخدام (13,785 عينة)
    ├── wlasl2000_dead.csv               # الروابط المفقودة أو المعطلة (7,191 عينة)
    └── splits/                          # تقسيمات التدريب والتحقق والاختبار الصالحة
        ├── wlasl2000_train_ok.csv       # مجموعة التدريب (9,211 عينة)
        ├── wlasl2000_val_ok.csv         # مجموعة التحقق (2,456 عينة)
        └── wlasl2000_test_ok.csv        # مجموعة الاختبار (2,118 عينة)
```

---

## 📊 ملخص نتائج فحص الروابط (Link Check Results)

- **إجمالي العينات المفحوصة:** 20,976 عينة.
- **الروابط الصالحة والمستجيبة (`ok`):** **13,785** عينة ($\approx 65.7\%$).
- **الروابط المعطلة (`dead`):** **7,191** عينة ($\approx 34.3\%$).
- **الفئات المغطاة (Glosses):** تم الحفاظ على تمثيل الفئات ضمن العينات الصالحة.

للتفاصيل الكاملة راجع: [`docs/wlasl_link_check_report.md`](docs/wlasl_link_check_report.md).

---

## 🚀 كيفية الاستخدام (Usage)

### 1. تنظيف الميتاداتا
```bash
python wlasl/scripts/clean_metadata.py --json WLASL_v0.3.json --out_dir wlasl/data
```

### 2. فحص صلاحية الروابط
```bash
python wlasl/scripts/check_links.py --csv wlasl/data/wlasl2000.csv --out_dir wlasl/data
```

### 3. استخراج نقاط الـ Keypoints (MediaPipe Holistic)
```bash
python wlasl/scripts/extract_keypoints.py --csv wlasl/data/wlasl2000_ok.csv --out_dir wlasl/kp2000
```
