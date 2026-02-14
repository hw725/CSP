# Datasets

Gold-standard test data for evaluating P2S and S2P pipeline accuracy.

## Structure

```
datasets/
└── splits/
    ├── paragraph_test.xlsx   # P2S evaluation (paragraph → sentence gold pairs)
    ├── sentence_test.xlsx    # S2P input / P2S output gold standard
    └── phrase_test.xlsx      # S2P evaluation (sentence → phrase gold pairs)
```

> **Note**: Training and validation splits (`*_train.xlsx`, `*_val.xlsx`) are generated locally and excluded from version control via `.gitignore`.

## Data Format

### paragraph_test.xlsx (P2S Input)

| Column | Description |
|--------|-------------|
| `para_id` | Paragraph identifier |
| `book_title` | Source book name |
| `source` | Classical Chinese paragraph (漢文 원문) |
| `target` | Korean translation paragraph (번역문) |

### sentence_test.xlsx (P2S Gold / S2P Input)

| Column | Description |
|--------|-------------|
| `para_id` | Parent paragraph identifier |
| `sent_id` | Sentence identifier within paragraph |
| `source` | Classical Chinese sentence |
| `target` | Korean translation sentence |

### phrase_test.xlsx (S2P Gold)

| Column | Description |
|--------|-------------|
| `sent_id` | Parent sentence identifier |
| `phrase_id` | Phrase identifier within sentence |
| `source` | Classical Chinese phrase |
| `target` | Korean translation phrase |

## Source Corpus

The full corpus consists of **44 classical Korean texts** (자치통감강목, 춘추좌씨전, 당시삼백수, etc.) stored locally in `xlsx/`. These are not committed to the repository due to size.

Total scale: **4,934 paragraphs** tested for P2S, **446 sentences** tested for S2P (v2.1).
