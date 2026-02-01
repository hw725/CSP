?# ?⑥궗(?룩쒔) 由ъ꽌移??섍꼍 援ъ텞 媛?대뱶

**?곌뎄 二쇱젣**: ?꾧렐? ?쒕Ц?낅쾿 醫낃껐?대????섎?濡좎쟻 湲곕뒫 寃利? 
**?곗씠?곗뀑**: ?꾪넗???ъ꽌(?쎿쎑) + ?쇨꼍(訝됬텚) 踰덉뿭 肄뷀띁??(364,007嫄?

---

## 1. ?섍꼍 ?붽뎄?ы빆

### Python
```
Python >= 3.10
```

### ?꾩닔 ?⑦궎吏
```bash
pip install pandas numpy scipy tqdm openai
```

### OpenAI API
- **紐⑤뜽**: `gpt-5-nano` (鍮꾩슜 ?⑥쑉???꾩닔議곗궗??
- **?섍꼍 蹂??*: `OPENAI_API_KEY` ?ㅼ젙 ?꾩슂

```bash
# Windows
set OPENAI_API_KEY=sk-...

# Linux/Mac
export OPENAI_API_KEY=sk-...
```

---

## 2. ?곗씠?곗뀑

### ?곗씠??異쒖쿂
?먮낯 ?곗씠?곕뒗 [?숈뼇怨좎쟾醫낇빀DB](https://db.juntong.or.kr)?먯꽌 ?遺遺?怨듦컻?섏뼱 ?덉뒿?덈떎.

### ?듬챸??泥섎━
踰덉뿭臾?以??쇰?????묎텒 蹂댄샇媛 ?꾩슂?섏뿬, 蹂???μ냼?먯꽌??**踰덉뿭臾몄쓣 SHA-256 ?댁떆濡?泥섎━**??踰꾩쟾???쒓났?⑸땲??

| ?뚯씪 | ?ㅻ챸 |
|------|------|
| `datasets/phrase_normalized_anonymized.csv` | 援??⑥쐞 ?곗씠??(踰덉뿭臾??댁떆 + LLM ?먯젙 寃곌낵 ?ы븿) |

### ?ы쁽??

#### ?ы쁽 媛?ν븳 寃?
- **?듦퀎 遺꾩꽍**: LLM ?먯젙 寃곌낵(O/X)媛 ?곗씠?곗뿉 ?ы븿?섏뼱 ?덉뼱 ?짼 寃?? p-value ???듦퀎 寃곌낵 ?ы쁽 媛??
- **留덉빱 鍮덈룄 遺꾩꽍**: ?먮Ц(?쒕Ц), 留덉빱, 硫뷀??곗씠?곕뒗 ?먮낯 洹몃?濡??좎?

#### ?ы쁽 遺덇??ν븳 寃?
- **LLM ?먯젙 怨쇱젙 ?먯껜**: 踰덉뿭臾??먮Ц???댁떆 泥섎━?섏뼱 ?덉뼱 LLM???ㅼ떆 吏덉쓽?섎뒗 寃껋? 遺덇???

#### ?щ챸???뺣낫
- **?꾨＼?꾪듃**: ?ъ슜??LLM ?꾨＼?꾪듃 ?꾨Ц 怨듦컻 (`dansa_full_survey.py`)
- **肄붾뱶**: 遺꾩꽍 ?뚯씠?꾨씪???꾩껜 怨듦컻
- **諛⑸쾿濡?*: 媛?? 寃利??덉감, ?듦퀎 湲곕쾿 ?곸꽭 臾몄꽌??
- **?먮낯 ?곗씠??*: 踰덉뿭臾??먮Ц???꾩슂??寃쎌슦 [?숈뼇怨좎쟾醫낇빀DB](https://db.juntong.or.kr)?먯꽌 ID濡?吏곸젒 議고쉶 媛??

---

## 3. ?듭떖 ?ㅽ겕由쏀듃

### Level 1-2 ?꾩닔議곗궗 (LLM 湲곕컲)
```bash
python dansa_full_survey.py
```
- Level 1: ?좎궗?대떒 '濡쒕떎' vs '?? (媛먰깂/?ъ슫)
- Level 2: 苡뚯젅/誘몄젅 '?덈씪' vs '?? (?⑦샇??醫낃껐)

### Level 3: 湲곗궗吏??'?섎떎' ?λⅤ 遺꾩꽍
```bash
python analyze_hada_by_genre.py
```

### 留덉빱 遺꾨쪟 諛??뺢퇋??
```bash
python phase4_premodern_classify.py
python hyeonto_normalizer.py
```

### 6?④퀎 ?⑥궗 寃利?
```bash
python verify_dansa_6levels.py
```

---

## 4. 寃곌낵臾?

### 蹂닿퀬??(`reports/phase4/`)
| ?뚯씪 | ?ㅻ챸 |
|------|------|
| `dansa_full_survey.json` | LLM ?꾩닔議곗궗 寃곌낵 |
| `hada_genre_analysis.json` | '?섎떎' ?λⅤ蹂?遺꾩꽍 |
| `PREMODERN_CLASSIFICATION.md` | ?꾧렐? 留덉빱 遺꾨쪟 泥닿퀎 |
| `all_markers_frequency.csv` | ?꾩껜 留덉빱 鍮덈룄??|


---

## 5. ?곌뎄 媛??

### ?⑥궗(?룩쒔) 6?④퀎 泥닿퀎 (?꾧퇋吏??딄뎄?먰빐踰뺛?

| ?④퀎 | 紐낆묶 | ?꾪넗 留덉빱 | ?섎? |
|:---:|------|----------|------|
| 1 | ?좎궗?대떒??窈욂세餓ζ뼴渦? | `-濡쒕떎` | 媛먰깂, ?ъ슫 |
| 2 | 苡뚯젅吏?⑥궗(鸚х독阿뗦뼴渦? | `-?대땲?? | ?⑦샇??醫낃껐 |
| 3 | 湲곗궗吏?⑥궗(鼇섆틟阿뗦뼴渦? | `-?섎떎` | 怨듭쟻 湲곕줉泥?|
| 4 | ?쒖닠吏?⑥궗(?띹염阿뗦뼴渦? | `-?섎뜑?? | ?쒖닠/?댁빞湲곗껜 |
| 5 | 誘몄젅吏?⑥궗(?ょ독阿뗦뼴渦? | `-?대씪` | 遺?쒕윭??醫낃껐 |
| 6 | ?몄슜?⑥궗(凉뺟뵪?룩쒔) | `-???섎떎` | ?몄슜 醫낃껐 |

---

## 6. ?ы쁽 ?덉감

1. **?섍꼍 ?ㅼ젙**
   ```bash
   pip install -r requirements.txt
   export OPENAI_API_KEY=sk-...
   ```

2. **?곗씠??以鍮?*
   - ?듬챸???곗씠?곕줈 ?듦퀎 遺꾩꽍 ?ы쁽 媛??
   - LLM ?먯젙 寃곌낵媛 ?곗씠?곗뿉 ?ы븿?섏뼱 ?덉쓬

3. **遺꾩꽍 ?ㅽ뻾**
   ```bash
   python dansa_full_survey.py
   ```

4. **寃곌낵 ?뺤씤**
   - `reports/phase4/dansa_full_survey.json`

---

## 7. ?몄슜

蹂??곌뎄???ㅼ쓬 臾명뿄??湲곕컲?⑸땲??

- ?꾧퇋吏? ?딄뎄?먰빐踰??θ?鰲ｆ퀡)??
- ?댁궪?? ?딄뎄?먯????θ??뉐뜔)??
- 諛뺣Ц?? ?딆씠?먰빐(岳싪?鰲???

---

## 8. ?쇱씠?좎뒪

- **肄붾뱶**: MIT License
- **?곗씠??*: ?곌뎄 紐⑹쟻 ?쒖젙 (?곸뾽???ъ슜 遺덇?)

---

*Last Updated: 2026-01-31*
 *cascade08*cascade08	 *cascade08	*cascade08 *cascade08*cascade08, *cascade08,/*cascade08/8 *cascade088;*cascade08;I *cascade08IL*cascade08LO *cascade08OR*cascade08RS *cascade08SV*cascade08VY *cascade08Y\*cascade08\f *cascade08fv*cascade08v? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08? *cascade08?*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08? *cascade08?*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08??*cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08? *cascade08?*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?*cascade08? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08?
*cascade08
? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08?? *cascade08??*cascade08??*cascade08?? *cascade08?*cascade08? *cascade08??*cascade08?? *cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?*cascade08? *cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?*cascade08? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08?? *cascade08??*cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08? *cascade08?*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08"(c9bfe0b46f5f7097b29a8f99d3ee94a0e38df5c92Lfile:///c:/Users/junto/Downloads/head-repo/hw725/CSP/hyeonto/DANSA_README.md:4file:///c:/Users/junto/Downloads/head-repo/hw725/CSP