# minbpe - BPE 학습 가이드

이 프로젝트는 LLM 토크나이저의 핵심인 **Byte Pair Encoding (BPE)** 알고리즘을 직접 구현하며 배우기 위한 교육용 저장소입니다.

## 왜 BPE를 배워야 할까요?

토크나이제이션은 LLM의 많은 "이상한" 동작들의 근본 원인입니다:
- LLM이 간단한 산술 연산을 틀리는 이유
- 영어가 아닌 언어에서 성능이 떨어지는 이유
- 문자열 뒤집기 같은 단순 작업을 못하는 이유

BPE를 이해하면 이런 현상들이 왜 발생하는지 명확히 알 수 있습니다.

---

## 학습 로드맵

### Phase 1: 기초 이해 (1-2일)

#### 1.1 이론 학습
1. **강의 영상 시청**: [Andrej Karpathy의 토크나이저 강의](https://www.youtube.com/watch?v=zduSFxRajkE)
2. **강의 문서 읽기**: [docs/lecture.md](docs/lecture.md) (한국어) 또는 [docs/lecture_origin.md](docs/lecture_origin.md) (영어)

#### 1.2 핵심 개념 체크리스트
- [ ] 문자 단위 토크나이제이션 vs 서브워드 토크나이제이션 차이
- [ ] BPE 알고리즘의 기본 원리 (가장 빈번한 쌍 병합)
- [ ] 왜 256개의 바이트 토큰으로 시작하는지
- [ ] vocab_size의 의미와 트레이드오프

### Phase 2: 코드 분석 (2-3일)

#### 2.1 코드 읽기 순서
```
1. minbpe/base.py     → 기본 구조 이해
2. minbpe/basic.py    → 핵심 BPE 구현
3. minbpe/regex.py    → GPT 스타일 전처리
4. minbpe/gpt4.py     → 실제 GPT-4 토크나이저 재현
```

#### 2.2 각 파일에서 집중해야 할 부분
| 파일 | 핵심 함수 | 이해해야 할 것 |
|------|----------|---------------|
| base.py | `save()`, `load()` | 토크나이저 저장 형식 |
| basic.py | `train()`, `encode()`, `decode()` | BPE의 핵심 로직 |
| regex.py | 정규식 패턴 | 왜 텍스트를 미리 분리하는지 |
| gpt4.py | `recover_merges()` | tiktoken과의 호환성 |

### Phase 3: 직접 구현 (3-5일)

[docs/exercise.md](docs/exercise.md)의 단계별 연습을 따라하세요:

| Step | 목표 | 난이도 |
|------|------|--------|
| Step 1 | BasicTokenizer 구현 | ★★☆☆☆ |
| Step 2 | RegexTokenizer로 확장 | ★★★☆☆ |
| Step 3 | GPT-4 토크나이저 매칭 | ★★★★☆ |
| Step 4 | 스페셜 토큰 처리 | ★★★★☆ |
| Step 5 | SentencePiece 스타일 구현 | ★★★★★ |

### Phase 4: 심화 학습

#### 4.1 실험해볼 것들
- 다양한 vocab_size로 학습하고 결과 비교
- 한국어 텍스트로 토크나이저 학습해보기
- 토큰화 결과가 모델 성능에 미치는 영향 분석

#### 4.2 토크나이저 시각화 도구
- [tiktokenizer](https://tiktokenizer.vercel.app) - 실시간 토크나이제이션 확인

---

## 후속 학습 자료

### 필수 논문

| 논문 | 핵심 내용 | 링크 |
|------|----------|------|
| **Sennrich et al. (2015)** | NLP에서 BPE 최초 적용 | [arXiv](https://arxiv.org/abs/1508.07909) |
| **GPT-2 (2019)** | Byte-level BPE 대중화 | [PDF](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |

### 추천 논문 (읽기 순서대로)

#### 토크나이제이션 기초
1. **"Neural Machine Translation of Rare Words with Subword Units"** (Sennrich et al., 2015)
   - BPE를 NLP에 처음 적용한 논문
   - 희귀 단어 문제를 서브워드로 해결하는 아이디어

2. **"SentencePiece: A simple and language independent subword tokenizer"** (Kudo & Richardson, 2018)
   - Google의 토크나이저 라이브러리
   - Llama, Mistral 등이 사용
   - [arXiv](https://arxiv.org/abs/1808.06226)

#### 토크나이제이션 개선
3. **"BPE-Dropout"** (Provilkov et al., 2020)
   - 학습 시 토크나이제이션에 노이즈 추가
   - 모델의 일반화 성능 향상
   - [arXiv](https://arxiv.org/abs/1910.13267)

4. **"UnigramLM"** (Kudo, 2018)
   - BPE의 대안적 접근법
   - 확률 기반 토크나이제이션
   - [arXiv](https://arxiv.org/abs/1804.10959)

#### 최신 연구
5. **"Tokenizer Choice For LLM Training"** (2024 연구들)
   - vocab size와 모델 성능의 관계
   - 다국어 토크나이저 설계

### 관련 코드베이스
- [tiktoken](https://github.com/openai/tiktoken) - OpenAI 공식 토크나이저
- [sentencepiece](https://github.com/google/sentencepiece) - Google 토크나이저
- [tokenizers](https://github.com/huggingface/tokenizers) - HuggingFace Rust 구현

---

## 프로젝트 구조

```
minbpe/
├── readme.md              # 이 파일 (학습 가이드)
├── train.py               # 토크나이저 학습 스크립트
├── minbpe/
│   ├── base.py            # 베이스 클래스
│   ├── basic.py           # 기본 BPE 구현
│   ├── regex.py           # 정규식 기반 토크나이저
│   └── gpt4.py            # GPT-4 토크나이저 래퍼
├── docs/
│   ├── readme.md          # 프로젝트 상세 문서 (한국어)
│   ├── readme_origin.md   # 프로젝트 상세 문서 (영어)
│   ├── lecture.md         # 강의 노트 (한국어)
│   ├── lecture_origin.md  # 강의 노트 (영어)
│   └── exercise.md        # 단계별 연습문제
└── tests/
    └── taylorswift.txt    # 학습용 샘플 텍스트
```

---

## 빠른 시작

```python
from minbpe import BasicTokenizer

# 토크나이저 생성 및 학습
tokenizer = BasicTokenizer()
tokenizer.train("your training text here", vocab_size=512)

# 인코딩/디코딩
tokens = tokenizer.encode("hello world")
text = tokenizer.decode(tokens)
```

자세한 사용법은 [docs/readme.md](docs/readme.md)를 참고하세요.

---

## 라이선스

MIT
