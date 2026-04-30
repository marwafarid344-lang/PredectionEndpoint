Ran command: `python -m uvicorn app:app --reload`

الموديل SVM بتاعك متدرب من غير `probability=True`، فمش بيدعم `predict_proba`. بس نقدر نحول الـ `decision_score` لـ confidence percentage باستخدام sigmoid function.

Viewed app.py:125-145
Edited app.py

الـ `--reload` هيعمل restart تلقائي. جرب تاني نفس الريكويست — دلوقتي الـ response هيبقى كده:

```json
{
  "prediction": 1,
  "label": "Heart Disease",
  "confidence": "82.3%",
  "probabilities": {
    "no_heart_disease": 17.7,
    "heart_disease": 82.3
  }
}
```

بستخدم **sigmoid function** عشان أحول الـ `decision_score` (1.5387) لنسبة مئوية مفهومة. جربها! 🚀
