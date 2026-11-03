"""BM25 용 한국어 형태소 분석: Kiwi 로 조사 · 어미를 떼고 명사 · 어간 · 외국어 · 숫자만 남긴다."""

from __future__ import annotations

from collections.abc import Sequence

from kiwipiepy import Kiwi, Token

# 검색어로 쓸 품사. 조사(J*) · 어미(E*) · 의존 명사(NNB) · 부호(S*) 는 뺀다
#   NNG 일반 명사 · NNP 고유 명사 · NR 수사 · NP 대명사 · XR 어근 · VV 동사 어간 · VA 형용사 어간
#   SL 외국어 · SH 한자 · SN 숫자
KEEP_TAGS = frozenset({"NNG", "NNP", "NR", "NP", "XR", "VV", "VA", "SL", "SH", "SN"})


class KiwiAnalyzer:
    """텍스트를 BM25 색인 · 질의용 토큰 목록으로 바꾼다.

    예)
        analyzer = KiwiAnalyzer.load()
        analyzer.tokens("서울특별시는 한양으로 불렸다")  # ["서울", "특별시", "한양", "불리"]

        kiwi = Kiwi()  # 같은 프로세스에서 Kiwi 를 따로 쓰는 쪽이 있으면 하나를 나눠 쓴다
        analyzer = KiwiAnalyzer(kiwi)
    """

    def __init__(self, kiwi: Kiwi) -> None:
        self._kiwi = kiwi

    @classmethod
    def load(cls) -> KiwiAnalyzer:
        """옵션 · 사용자 사전 없는 Kiwi 모델을 올려 만든다. 모델 하나가 메모리를 약 500MB 쓴다."""
        return cls(Kiwi())

    def tokens(self, text: str) -> list[str]:
        """한 텍스트의 검색 토큰. 외국어는 소문자로."""
        return _keep(self._kiwi.tokenize(text))

    def tokens_batch(self, texts: Sequence[str]) -> list[list[str]]:
        """여러 텍스트를 한꺼번에 분석한다 (Kiwi 가 여러 스레드로 처리). 입력 순서를 지킨다."""
        return [_keep(tokens) for tokens in self._kiwi.tokenize(list(texts))]


def _keep(tokens: Sequence[Token]) -> list[str]:
    return [token.form.lower() for token in tokens if token.tag in KEEP_TAGS]
