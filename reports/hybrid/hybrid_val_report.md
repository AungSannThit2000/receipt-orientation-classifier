# Hybrid OCR Evaluation - Val

- Samples: 360
- Model-only accuracy: 92.50%
- Pairwise OCR-only accuracy: 96.94%
- Hybrid accuracy: 97.22%
- OCR minimum score: 0.00
- OCR minimum margin: 0.22
- OCR confirmations: 316
- OCR overrides: 26
- OCR inconclusive fallbacks: 18
- Helpful overrides: 21
- Harmful overrides: 4

The model selects the vertical or horizontal orientation pair. OCR then compares only the two opposite directions in that pair. Weak or closely tied OCR results cannot override the model.
