?# Hyeonto ?곌뎄 ?ы쁽 媛?대뱶 (Reproduction Guide) - v6.9.4 Final

蹂?臾몄꽌??hyeonto ?꾨줈?앺듃???곗씠??媛?⑹꽦 ?뺣낫? ?꾩껜 遺꾩꽍 ?뚯씠?꾨씪?몄쓣 泥섏쓬遺???앷퉴吏 ?ы쁽?섎뒗 諛⑸쾿???④퀎蹂꾨줈 ?ㅻ챸?⑸땲??

> **?뱦 遺꾩꽍 ?⑥쐞**
> - **Sentence**: 臾몄옣 ?⑥쐞 ?대윭?ㅽ꽣留?(150,545嫄?
> - **Phrase**: 援??⑥쐞 ?대윭?ㅽ꽣留?(366,222嫄?

---

## ?뱛 ?곗씠??媛?⑹꽦 諛??덈궡 (Data Availability)

### 1. ?곗씠??異쒖쿂

蹂??곌뎄???ъ슜???꾪넗 ?곗씠?곕뒗 **?숈뼇怨좎쟾醫낇빀DB**(https://db.juntong.or.kr)?먯꽌 ?쒓났?섎뒗 ?ъ꽌?쇨꼍 諛?湲고? ?좉탳 寃쎌쟾???꾪넗蹂몄쓣 湲곕컲?쇰줈 ?⑸땲??

### 2. ??묎텒 諛??ы쁽???덈궡

?좑툘 **?쇰? ?띿뒪?몃뒗 ??묎텒 臾몄젣濡?怨듦컻 ?묎렐???쒗븳?????덉뒿?덈떎.**

洹몃윭??怨듦컻???띿뒪?몃쭔?쇰줈??**蹂??곌뎄???듭떖 諛쒓껄??異⑸텇???ы쁽 媛??*?⑸땲??

### 3. ?곗씠???ㅽ궎留?(CSV)

遺꾩꽍???곗씠?곗뀑??援ъ“:

| 而щ읆紐?| ?ㅻ챸 | ?덉떆 |
|:---|:---|:---|
| `臾몃떒?앸퀎?? | 臾몃떒 怨좎쑀 ID | 1 |
| `臾몄옣?앸퀎?? | 臾몄옣 ?쒕쾲 | 2 |
| `?먮Ц` | ?꾪넗媛 ?ы븿???쒕Ц ?먮Ц | 耶붷춴???띴툡??耶쀤뺘弱쇰땲 ?뜹뀍? 若뗤볶?대씪 |
| `踰덉뿭臾? | ?쒓뎅??踰덉뿭臾?| 怨듭옄(耶붷춴)???대쫫??援?訝???.. |
| `book_name` | ?꾩꽌紐?| ?쇱뼱吏묒＜ |

---

## ?뱥 ?ъ쟾 以鍮?(Prerequisites)

### 1. ?섍꼍 ?ㅼ젙

**?꾩슂???뚰봽?몄썾??*:
- Python 3.9 ?댁긽
- CUDA 11.8 ?댁긽 (GPU ?ъ슜 ??
- Docker (沅뚯옣 - `csp-workspace` 而⑦뀒?대꼫)

**?꾩닔 ?⑦궎吏**:
```bash
pip install regex  # Unicode Script Property 吏??
```

### 2. ?곗씠??諛곗튂

- XML ?먮낯: `hyeonto/*.xml`
- ?듯빀 CSV: `hyeonto/datasets/sentence_merged_v2.csv`, `hyeonto/datasets/phrase_merged_v2.csv`

---

## ?봽 ?꾩껜 ?뚯씠?꾨씪???ㅽ뻾 (v6.9.4)

### Phase 0: Production ?뚯씠?꾨씪??(沅뚯옣)

**v6.9.4?먯꽌??`run_full_pipeline.py`媛 怨듭떇 ?뚯씠?꾨씪?몄엯?덈떎.**

```bash
# ?꾩껜 ?뚯씠?꾨씪???ㅽ뻾 (from scratch)
docker compose run --rm csp python hyeonto/run_full_pipeline.py
```

???ㅽ겕由쏀듃???ㅼ쓬???섑뻾?⑸땲??
1. XML ?먮낯?먯꽌 ?곗씠??異붿텧
2. 171媛?留덉빱 ?뺢퇋??(Zero-Gap)
3. `\p{Hangul}+` Unicode Regex濡??쏇븳湲 ?ъ갑
4. BGE-M3 ?꾨쿋???앹꽦 諛?罹먯떆
5. K=4 ?대윭?ㅽ꽣留?諛??꾨줈?뚯씪留?

---

### Phase 1: ?꾨쿋??罹먯떆 ?앹꽦 (媛쒕퀎 ?ㅽ뻾 ??

?洹쒕え Phrase ?곗씠????36留?嫄????꾨쿋??怨꾩궛 ?쒓컙???덉빟?섍린 ?꾪빐 罹먯떆瑜??앹꽦?⑸땲??

```bash
# Sentence ?꾨쿋??罹먯떆 ?앹꽦 (??30遺?
docker exec csp-workspace python scripts/cluster_pa_boundary_functions.py \
    --input hyeonto/datasets/sentence_merged_v2.csv \
    --out-dir hyeonto/reports/temp \
    --k 4 --use-src --use-tgt \
    --save-embeddings hyeonto/cache/sentence_embeddings.npy \
    --device-id 0 --seed 42

# Phrase ?꾨쿋??罹먯떆 ?앹꽦 (??5-6?쒓컙, resume 吏??
docker exec csp-workspace python scripts/find_optimal_k.py \
    --csv hyeonto/datasets/phrase_merged_v2.csv \
    --out-dir hyeonto/reports/optimal_k_phrase \
    --k-min 4 --k-max 32 --k-step 2 \
    --save-embeddings hyeonto/cache/phrase_embeddings.npy \
    --device-id 0 --seed 42
```

> **李멸퀬**: 以묐떒 ??`hyeonto/cache/phrase_embeddings_resume.npy`??以묎컙 寃곌낵媛 ??λ릺硫? ?ъ떆?????먮룞?쇰줈 ?댁뼱??吏꾪뻾?⑸땲??

---

### Phase 2: Sentence ?대윭?ㅽ꽣留?(K=4)

```bash
docker exec csp-workspace python scripts/cluster_pa_boundary_functions.py \
    --input hyeonto/datasets/sentence_merged_v2.csv \
    --out-dir hyeonto/reports/sentence_k4_normalized \
    --k 4 --load-embeddings hyeonto/cache/sentence_embeddings.npy \
    --use-src --use-tgt --seed 42 --max-boundaries 500000
```

---

### Phase 3: Phrase ?대윭?ㅽ꽣留?(K=4)

```bash
docker exec csp-workspace python scripts/cluster_sa_boundary_functions.py \
    --input hyeonto/datasets/phrase_merged_v2.csv \
    --out-dir hyeonto/reports/phrase_k4_normalized \
    --k 4 --load-embeddings hyeonto/cache/phrase_embeddings.npy \
    --use-src --use-tgt --seed 42 --max-boundaries 500000
```

---

### Phase 4: ?꾨줈?뚯씪留?

```bash
# Sentence K=4 ?꾨줈?뚯씪留?
docker exec csp-workspace python scripts/profile_boundary_clusters.py \
    --csv hyeonto/reports/sentence_k4_normalized/boundary_clusters.csv \
    --out hyeonto/reports/sentence_k4_normalized/sentence_cluster_profile.md

# Phrase K=4 ?꾨줈?뚯씪留?
docker exec csp-workspace python scripts/profile_boundary_clusters.py \
    --csv hyeonto/reports/phrase_k4_normalized/sa_boundary_clusters.csv \
    --out hyeonto/reports/phrase_k4_normalized/phrase_cluster_profile.md
```

---

### Phase 5: ?쒓컖??

```bash
# UMAP 3D/2D ?쒓컖??(Docker GPU ?섍꼍)
docker compose run --rm csp python hyeonto/analyze_embedding_overlay.py

# Sentence-Phrase Sankey ?ㅼ씠?닿렇??(K=4 ??K=4)
docker compose run --rm csp python hyeonto/generate_sankey_diagrams.py
```

---

## ?㎦ 寃곌낵 寃利?(Verification)

### 1. ?곗씠???섏튂 ?뺤씤 (v6.9.4)

| ?곗씠?곗뀑 | ?덉긽 ????(header ?쒖쇅) |
|:---|:---:|
| Sentence K=4 | 150,545 |
| Phrase K=4 | 366,222 |

### 2. 二쇱슂 吏???뺤씤

- **Sentence K=4 p1 Canonicity**: ??13.4%
- **Phrase K=4 p5 Canonicity**: ??16.5%
- **肄뷀띁??臾닿껐??*: `?뉕퀬` 359/178嫄? `?뉕?` 1,095/963嫄?(Sentence/Phrase)

### 3. 留덉빱 ?ㅽ궎留?寃利?

```bash
# ?뺢퇋??媛?遺꾩꽍 (Zero-Gap ?뺤씤)
python hyeonto/analyze_normalization_gaps.py
# ?덉긽 寃곌낵: "Zero additional candidates" for 171-entry schema
```

---

## ?깍툘 ?덉긽 ?뚯슂 ?쒓컙

| ?④퀎 | GPU (RTX 3090) | CPU |
|------|:---:|:---:|
| ?꾨쿋??罹먯떆 (Phrase 36留뚭굔) | ??5-6?쒓컙 | ??24?쒓컙+ |
| ?대윭?ㅽ꽣留?(罹먯떆 ?ъ슜) | ??2遺?K | ??5遺?K |
| ?꾨줈?뚯씪留?| ??5遺?| ??10遺?|
| ?쒓컖??| ??5遺?| ??10遺?|

---

## ?맀 臾몄젣 ?닿껐 (Troubleshooting)

- **GPU 硫붾え由?遺議?*: `--batch 64` (?먮뒗 ????쾶) ?듭뀡 異붽?
- **Phrase ?꾨쿋??以묐떒**: `hyeonto/cache/phrase_embeddings_resume.npy`?먯꽌 ?먮룞 ?ш컻
- **Docker 而⑦뀒?대꼫**: `docker start csp-workspace` 紐낅졊?쇰줈 ?뺤씤
- **?쏇븳湲 ?꾨씫**: `regex` ?쇱씠釉뚮윭由ъ? `\p{Hangul}+` ?⑦꽩 ?ъ슜 ?뺤씤

---

## ?뱥 湲곗닠 ?쒖? (v6.9.4)

### Unicode Regex ?쒖???

```python
import regex  # 'regex' ?쇱씠釉뚮윭由?(re ?꾨떂)

# ?쒓? (?쏇븳湲 ?ы븿)
HANGUL_PATTERN = r'\p{Hangul}+'

# ?쒖옄
HANJA_PATTERN = r'\p{Han}'
```

**?곸슜 ?ㅽ겕由쏀듃**:
- `analyze_normalization_gaps.py`
- `analyze_cooccurrence_normalized.py`
- `run_full_pipeline.py`

---

**?낅뜲?댄듃 ?쇱옄**: 2026-01-27
**?묒꽦??*: CSP Research Team
?"(c9bfe0b46f5f7097b29a8f99d3ee94a0e38df5c92Ifile:///c:/Users/junto/Downloads/head-repo/hw725/CSP/hyeonto/REPRODUCE.md:4file:///c:/Users/junto/Downloads/head-repo/hw725/CSP