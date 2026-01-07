# LLM 토크나이제이션

안녕하세요, 오늘은 대규모 언어 모델(LLM)에서의 토크나이제이션에 대해 알아보겠습니다. 안타깝게도 토크나이제이션은 최신 LLM에서 꽤 복잡하고 까다로운 부분입니다. 하지만 자세히 이해할 필요가 있는데, LLM의 여러 단점들이 신경망 문제처럼 보이거나 뭔가 이상해 보이는 것들이 사실은 토크나이제이션에서 비롯되는 경우가 많기 때문입니다.

### 이전 내용: 문자 단위 토크나이제이션

그래서 토크나이제이션이 뭘까요? 사실 이전 영상 [GPT를 처음부터 만들어보자](https://www.youtube.com/watch?v=kCc8FmEb1nY)에서 토크나이제이션을 다룬 적이 있는데, 그때는 아주 단순하고 naive한 문자 단위 버전이었습니다. 그 영상의 [Google colab](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)에 가보시면, 학습 데이터([셰익스피어](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt))로 시작했는데, 이건 그냥 Python의 큰 문자열입니다:

```
First Citizen: Before we proceed any further, hear me speak.

All: Speak, speak.

First Citizen: You are all resolved rather to die than to famish?

All: Resolved. resolved.

First Citizen: First, you know Caius Marcius is chief enemy to the people.

All: We know't, we know't.
```

근데 문자열을 언어 모델에 어떻게 넣을까요? 우리는 먼저 전체 학습 데이터에서 발견된 모든 가능한 문자들의 어휘를 구성했습니다:

```python
# 이 텍스트에 나타나는 모든 고유 문자들
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# 65
```

그리고 위의 어휘에 따라 개별 문자와 정수 간의 변환을 위한 룩업 테이블을 만들었습니다. 이 룩업 테이블은 그냥 Python 딕셔너리입니다:

```python
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
# encoder: 문자열을 받아서 정수 리스트 출력
encode = lambda s: [stoi[c] for c in s]
# decoder: 정수 리스트를 받아서 문자열 출력
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hii there"))
print(decode(encode("hii there")))

# [46, 47, 47, 1, 58, 46, 43, 56, 43]
# hii there
```

문자열을 정수 시퀀스로 변환하고 나면, 각 정수는 학습 가능한 파라미터의 2차원 임베딩에 대한 인덱스로 사용됩니다. 어휘 크기가 `vocab_size=65`이므로, 이 임베딩 테이블도 65개의 행을 가지게 됩니다:

```python
class BigramLanguageModel(nn.Module):

def __init__(self, vocab_size):
	super().__init__()
	self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

def forward(self, idx, targets=None):
	tok_emb = self.token_embedding_table(idx) # (B,T,C)
```

여기서 정수는 이 임베딩 테이블의 한 행을 "뽑아오고", 이 행이 해당 토큰을 나타내는 벡터가 됩니다. 이 벡터가 해당 타임스텝의 입력으로 Transformer에 들어갑니다.

### BPE 알고리즘을 사용한 "문자 청크" 토크나이제이션

문자 단위 언어 모델의 naive한 설정에서는 이게 다 괜찮습니다. 하지만 실제로 최신 언어 모델들에서는 토큰 어휘를 구성하는 데 훨씬 더 복잡한 방식을 사용합니다. 특히, 이런 방식들은 문자 단위가 아니라 문자 청크 단위로 동작합니다. 그리고 이러한 청크 어휘를 구성하는 방법이 바로 **Byte Pair Encoding** (BPE) 같은 알고리즘이고, 아래에서 자세히 다룰 예정입니다.

이 접근법의 역사적 발전을 잠깐 살펴보면, 언어 모델 토크나이제이션에 바이트 수준 BPE 알고리즘 사용을 대중화시킨 논문은 2019년 OpenAI의 [GPT-2 논문](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) "Language Models are Unsupervised Multitask Learners"입니다. 섹션 2.2 "Input Representation"으로 스크롤해보시면 이 알고리즘을 설명하고 있습니다. 이 섹션 끝에서 다음과 같이 말합니다:

> *어휘가 50,257로 확장되었습니다. 또한 컨텍스트 크기를 512에서 1024 토큰으로 늘리고, 배치 크기도 512로 더 크게 사용했습니다.*

Transformer의 어텐션 레이어에서 모든 토큰은 시퀀스에서 이전에 있는 유한한 토큰 목록에 attend한다는 것을 기억하세요. 여기서 논문은 GPT-2 모델이 GPT-1의 512에서 늘어난 1024 토큰의 컨텍스트 길이를 가진다고 말합니다. 다시 말해, 토큰은 LLM 입력의 기본적인 "원자"입니다. 그리고 토크나이제이션은 Python의 원시 문자열을 토큰 리스트로 변환하고, 그 반대도 수행하는 과정입니다. 이 추상화가 얼마나 널리 쓰이는지 보여주는 또 다른 예로, [Llama 2](https://arxiv.org/abs/2307.09288) 논문에서 "token"을 검색하면 63번 나옵니다. 예를 들어, 논문에서는 2조 개의 토큰으로 학습했다고 합니다.

### 토크나이제이션의 복잡성 맛보기

구현의 세부사항으로 들어가기 전에, 토크나이제이션 과정을 자세히 이해해야 하는 이유를 간단히 알아봅시다. 토크나이제이션은 LLM의 많은 이상한 현상들의 핵심에 있고, 대충 넘기지 않으시길 권합니다. 신경망 아키텍처 문제처럼 보이는 많은 이슈들이 실제로는 토크나이제이션으로 거슬러 올라갑니다. 몇 가지 예를 들어보면:

- LLM이 왜 단어 철자를 못 맞출까요? **토크나이제이션**.
- LLM이 왜 문자열 뒤집기 같은 아주 간단한 문자열 처리를 못할까요? **토크나이제이션**.
- LLM이 왜 비영어권 언어(예: 일본어)에서 성능이 떨어질까요? **토크나이제이션**.
- LLM이 왜 간단한 산술 연산을 못할까요? **토크나이제이션**.
- GPT-2가 왜 Python 코딩에서 필요 이상으로 어려움을 겪었을까요? **토크나이제이션**.
- 제 LLM이 왜 "<|endoftext|>" 문자열을 보면 갑자기 멈출까요? **토크나이제이션**.
- "trailing whitespace"에 대한 이상한 경고가 뭘까요? **토크나이제이션**.
- "SolidGoldMagikarp"에 대해 물어보면 왜 LLM이 망가질까요? **토크나이제이션**.
- LLM에서 JSON보다 YAML을 선호해야 하는 이유는? **토크나이제이션**.
- LLM이 왜 진정한 end-to-end 언어 모델링이 아닐까요? **토크나이제이션**.
- 고통의 진짜 근원은 무엇일까요? **토크나이제이션**.

영상 끝에서 이것들을 다시 다룰 예정입니다.

### 토크나이제이션 시각적 미리보기

다음으로, 이 [토크나이제이션 웹앱](https://tiktokenizer.vercel.app)을 로드해봅시다. 이 웹앱의 좋은 점은 토크나이제이션이 웹 브라우저에서 실시간으로 실행되어서, 입력에 텍스트 문자열을 쉽게 넣고 오른쪽에서 토크나이제이션 결과를 볼 수 있다는 점입니다. 상단에서 현재 `gpt2` 토크나이저를 사용하고 있다는 것을 볼 수 있고, 이 예시에서 붙여넣은 문자열이 현재 300개의 토큰으로 토크나이징된다는 것을 볼 수 있습니다. 여기서 색상으로 명시적으로 보여집니다:

![tiktokenizer](assets/tiktokenizer.png)

예를 들어, "Tokenization" 문자열은 토큰 30642와 토큰 1634로 인코딩됩니다. " is" 토큰(이게 세 글자라는 점에 주목하세요, 앞에 공백 포함이고, 이게 중요합니다!)은 인덱스 318입니다. 공백에 주의하세요. 공백은 분명히 문자열에 존재하고 다른 모든 문자들과 함께 토크나이징되어야 하지만, 보통 시각화에서는 명확성을 위해 생략됩니다. 앱 하단에서 시각화를 켜고 끌 수 있습니다. 같은 방식으로, " at" 토큰은 379, " the"는 262 등입니다.

다음으로, 간단한 산술 예시가 있습니다. 여기서 숫자들이 토크나이저에 의해 일관성 없이 분해될 수 있다는 것을 볼 수 있습니다. 예를 들어, 숫자 127은 세 글자의 단일 토큰이지만, 숫자 677은 두 개의 토큰이 됩니다: " 6" 토큰(다시 말하지만, 앞에 공백 주목!)과 "77" 토큰. 우리는 대규모 언어 모델이 이런 임의성을 이해하도록 의존합니다. 모델은 학습 중에 파라미터 안에서 이 두 토큰(" 6"과 "77"이 실제로 결합해서 숫자 677을 만든다는 것)을 배워야 합니다. 마찬가지로, LLM이 이 합의 결과가 804라고 예측하려면, 두 타임스텝에 걸쳐 출력해야 합니다: 먼저 " 8" 토큰을 내보내고, 그다음 "04" 토큰을 내보내야 합니다. 이 모든 분할이 완전히 임의적으로 보인다는 점에 주목하세요. 바로 아래 예시에서, 1275는 "12" 다음에 "75"이고, 6773은 실제로 두 토큰 " 6", "773"이고, 8041은 " 8", "041"입니다.

(계속...)
(TODO: 영상에서 자동으로 생성하는 방법을 알아내지 않는 한 이어서 작성할 수도 있습니다 :))
