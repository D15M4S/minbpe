# minbpe

LLM 토크나이저에서 자주 쓰이는 (byte-level) Byte Pair Encoding (BPE) 알고리즘을 깔끔하고 미니멀하게 구현한 코드입니다. "byte-level"이라고 하는 이유는 UTF-8로 인코딩된 문자열에서 동작하기 때문입니다.

이 알고리즘은 [GPT-2 논문](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)과 OpenAI의 [GPT-2 코드](https://github.com/openai/gpt-2)를 통해 LLM 분야에서 유명해졌습니다. NLP에서 BPE를 처음 사용한 논문은 [Sennrich et al. 2015](https://arxiv.org/abs/1508.07909)입니다. 요즘 나오는 LLM들(GPT, Llama, Mistral 등)은 거의 다 이 알고리즘으로 토크나이저를 학습시킵니다.

이 저장소에는 두 가지 Tokenizer가 있고, 둘 다 토크나이저의 3가지 핵심 기능을 수행할 수 있습니다: 1) 주어진 텍스트로 어휘와 병합 규칙 학습, 2) 텍스트를 토큰으로 인코딩, 3) 토큰을 텍스트로 디코딩. 파일 구성은 다음과 같습니다:

1. [minbpe/base.py](minbpe/base.py): `Tokenizer` 클래스를 구현합니다. 베이스 클래스로, `train`, `encode`, `decode` 스텁과 저장/로드 기능, 그리고 몇 가지 유틸리티 함수가 들어있습니다. 직접 사용하는게 아니라 상속받아서 쓰라고 만든 클래스입니다.
2. [minbpe/basic.py](minbpe/basic.py): `BasicTokenizer`를 구현합니다. 텍스트에서 바로 돌아가는 가장 단순한 BPE 구현체입니다.
3. [minbpe/regex.py](minbpe/regex.py): `RegexTokenizer`를 구현합니다. 토큰화 전에 정규식 패턴으로 입력 텍스트를 전처리해서 카테고리별로(문자, 숫자, 구두점 등) 나눕니다. 이렇게 하면 카테고리 경계를 넘어서 병합이 일어나지 않습니다. GPT-2 논문에서 도입됐고 GPT-4까지 계속 쓰이고 있습니다. 스페셜 토큰도 이 클래스에서 처리합니다.
4. [minbpe/gpt4.py](minbpe/gpt4.py): `GPT4Tokenizer`를 구현합니다. `RegexTokenizer`를 감싸는 가벼운 래퍼로, [tiktoken](https://github.com/openai/tiktoken) 라이브러리의 GPT-4 토크나이저를 정확히 재현합니다. 정확한 병합 규칙을 복원하는 것과 1바이트 토큰 순서 관련 처리(아마 역사적인 이유로 생긴 듯?)를 담당합니다.

마지막으로 [train.py](train.py) 스크립트는 [tests/taylorswift.txt](tests/taylorswift.txt) 텍스트(테일러 스위프트 위키백과 문서ㅋㅋ)로 두 가지 주요 토크나이저를 학습시키고 vocab을 디스크에 저장합니다. 제 맥북(M1)에서 약 25초 정도 걸립니다.

위의 모든 파일들은 짧고 주석도 잘 달려있고, 파일 하단에 사용 예시도 있습니다.

## 빠른 시작

가장 간단한 예시로, [위키백과 BPE 문서](https://en.wikipedia.org/wiki/Byte_pair_encoding)를 따라해볼 수 있습니다:

```python
from minbpe import BasicTokenizer
tokenizer = BasicTokenizer()
text = "aaabdaaabac"
tokenizer.train(text, 256 + 3) # 256개는 바이트 토큰이고, 그 다음 3번 병합
print(tokenizer.encode(text))
# [258, 100, 258, 97, 99]
print(tokenizer.decode([258, 100, 258, 97, 99]))
# aaabdaaabac
tokenizer.save("toy")
# 두 파일 생성: toy.model (로드용)과 toy.vocab (확인용)
```

위키백과에 따르면, "aaabdaaabac"에 BPE를 3번 병합하면 "XdXac"가 됩니다 (X=ZY, Y=ab, Z=aa). 참고로 minbpe는 항상 256개의 개별 바이트를 토큰으로 먼저 할당하고, 거기서부터 필요한 만큼 병합합니다. 그래서 a=97, b=98, c=99, d=100 ([ASCII](https://www.asciitable.com) 값)입니다. (a,a)가 Z로 병합되면 Z는 256이 되고, Y는 257, X는 258이 됩니다. 256개 바이트에서 시작해서 3번 병합하면 위의 결과 [258, 100, 258, 97, 99]가 나옵니다.

## 추론: GPT-4 비교

`RegexTokenizer`가 [tiktoken](https://github.com/openai/tiktoken)의 GPT-4 토크나이저와 동일하게 동작하는지 확인할 수 있습니다:

```python
text = "hello123!!!? (안녕하세요!) 😉"

# tiktoken
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode(text))
# [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]

# 우리 코드
from minbpe import GPT4Tokenizer
tokenizer = GPT4Tokenizer()
print(tokenizer.encode(text))
# [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]
```

(실행하려면 `pip install tiktoken` 필요합니다). 내부적으로 `GPT4Tokenizer`는 `RegexTokenizer`를 감싸서 GPT-4의 병합 규칙과 스페셜 토큰을 전달하는 것뿐입니다. 스페셜 토큰도 잘 처리되는지 확인할 수 있습니다:

```python
text = "<|endoftext|>hello world"

# tiktoken
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode(text, allowed_special="all"))
# [100257, 15339, 1917]

# 우리 코드
from minbpe import GPT4Tokenizer
tokenizer = GPT4Tokenizer()
print(tokenizer.encode(text, allowed_special="all"))
# [100257, 15339, 1917]
```

tiktoken처럼 encode 호출 시 스페셜 토큰 사용 의도를 명시적으로 선언해야 합니다. 안 그러면 공격자가 제어하는 데이터(예: 유저 프롬프트)에 스페셜 토큰이 들어가는 보안 문제가 생길 수 있습니다. `allowed_special` 파라미터는 "all", "none", 또는 허용할 스페셜 토큰 리스트로 설정할 수 있습니다.

## 학습

tiktoken과 달리 이 코드로 직접 토크나이저를 학습시킬 수 있습니다. 이론적으로 `RegexTokenizer`를 큰 데이터셋에서 vocab 크기 100K로 학습시키면 GPT-4 토크나이저를 재현할 수 있을 겁니다.

두 가지 방법이 있습니다. 첫째로, 정규식 패턴으로 텍스트를 분리하고 전처리하는 복잡함이 싫고, 스페셜 토큰도 필요 없다면 `BasicTokenizer`를 쓰면 됩니다:

```python
from minbpe import BasicTokenizer
tokenizer = BasicTokenizer()
tokenizer.train(very_long_training_string, vocab_size=4096)
tokenizer.encode("hello world") # 문자열 -> 토큰
tokenizer.decode([1000, 2000, 3000]) # 토큰 -> 문자열
tokenizer.save("mymodel") # mymodel.model과 mymodel.vocab 생성
tokenizer.load("mymodel.model") # 모델 로드, vocab은 시각화용
```

OpenAI 방식을 따라하고 싶다면 정규식 패턴으로 텍스트를 카테고리별로 나누는 방법을 쓰면 됩니다. GPT-4 패턴이 `RegexTokenizer`의 기본값이니까 이렇게 하면 됩니다:

```python
from minbpe import RegexTokenizer
tokenizer = RegexTokenizer()
tokenizer.train(very_long_training_string, vocab_size=32768)
tokenizer.encode("hello world") # 문자열 -> 토큰
tokenizer.decode([1000, 2000, 3000]) # 토큰 -> 문자열
tokenizer.save("tok32k") # tok32k.model과 tok32k.vocab 생성
tokenizer.load("tok32k.model") # 디스크에서 모델 로드
```

당연히 데이터셋 크기에 따라 vocab 크기를 조절하면 됩니다.

**스페셜 토큰**. 스페셜 토큰을 추가하고 싶으면 `register_special_tokens` 함수를 쓰면 됩니다. 예를 들어 vocab_size 32768로 학습했다면, 처음 256개는 바이트 토큰이고, 다음 32768-256개는 병합 토큰이고, 그 다음에 스페셜 토큰을 추가할 수 있습니다. 마지막 "진짜" 병합 토큰 id가 32767 (vocab_size - 1)이니까 첫 번째 스페셜 토큰은 그 바로 다음인 32768이어야 합니다:

```python
from minbpe import RegexTokenizer
tokenizer = RegexTokenizer()
tokenizer.train(very_long_training_string, vocab_size=32768)
tokenizer.register_special_tokens({"<|endoftext|>": 32768})
tokenizer.encode("<|endoftext|>hello world", allowed_special="all")
```

그 뒤에 토큰을 더 추가해도 됩니다. 마지막으로, 코드를 깔끔하고 읽기 쉽고 수정하기 쉽게 만들려고 노력했습니다. 코드 읽고 이해하는 거 겁먹지 마세요. 테스트 코드에도 사용 예시가 많으니까 참고하시면 좋습니다.

## 테스트

테스트는 pytest를 사용합니다. 테스트 파일들은 전부 `tests/` 디렉토리에 있습니다. `pip install pytest` 먼저 하고:

```bash
$ pytest -v .
```

이렇게 실행하면 됩니다. (-v는 verbose 옵션, 좀 더 보기 좋게 출력됨)

## 커뮤니티 확장

* [gnp/minbpe-rs](https://github.com/gnp/minbpe-rs): `minbpe`의 Rust 구현체. Python 버전과 거의 1:1로 대응됩니다.

## 연습문제

BPE를 공부하고 싶은 분들을 위해, 직접 minbpe를 단계별로 만들어볼 수 있는 연습문제가 있습니다. [exercise.md](exercise.md) 참고하세요.

## 강의

이 저장소의 코드를 [유튜브 영상](https://www.youtube.com/watch?v=zduSFxRajkE)에서 만들었습니다. 텍스트 버전은 [lecture.md](lecture.md)에서 볼 수 있습니다.

## 할 일

- 큰 파일과 큰 vocab에서도 돌아가는 최적화된 Python 버전 만들기
- 더 최적화된 C나 Rust 버전 만들기
- GPT4Tokenizer를 GPTTokenizer로 바꾸고 GPT-2/GPT-3/GPT-3.5도 지원?
- GPT4Tokenizer처럼 LlamaTokenizer 만들기 (sentencepiece 동등하게)

## 라이선스

MIT
