"""슬라이딩 윈도우 분할.

창 경계는 가드 자신의 토크나이저가 아니라 **고정된 외부 토크나이저**의 토큰 offset 으로
잡는다. 가드마다 토크나이저가 달라, 각자의 토크나이저를 쓰면 "32 토큰"이 모델마다 다른
길이를 뜻하게 되고 창 크기를 가드끼리 비교할 수 없다.
"""

from __future__ import annotations

from functools import lru_cache

#: 창 경계를 잡는 데만 쓰는 토크나이저. 작고 빠르며 특정 가드에 매이지 않는다.
BOUNDARY_TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def _load(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def split_windows(
    text: str,
    window: int = 64,
    stride: int | None = None,
    tokenizer: str = BOUNDARY_TOKENIZER,
) -> list[str]:
    """`text` 를 `window` 토큰 길이의 겹치는 창으로 자른다.

    `stride` 를 생략하면 창의 절반이 된다. 마지막 창은 텍스트 끝에 붙여 꼬리가 잘리지
    않게 한다. 텍스트가 창 하나보다 짧으면 통째로 한 조각을 돌려준다.
    """
    stride = stride or max(1, window // 2)
    tok = _load(tokenizer)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False,
              truncation=True, max_length=8192)
    off = [(a, b) for a, b in enc["offset_mapping"] if b > a]
    if not off:
        return [text]

    step = max(1, min(stride, window))
    starts = list(range(0, max(len(off) - window, 0) + 1, step))
    if starts[-1] + window < len(off):
        starts.append(len(off) - window)          # 꼬리를 남긴다
    pieces = [text[off[i][0]:off[min(i + window, len(off)) - 1][1]].strip()
              for i in starts]
    return [p for p in pieces if p] or [text]


def split_many(texts: list[str], window: int = 64, stride: int | None = None,
               tokenizer: str = BOUNDARY_TOKENIZER) -> list[list[str]]:
    """여러 텍스트에 `split_windows` 를 적용한다. 입력마다 조각 목록을 하나씩 돌려준다."""
    return [split_windows(t, window, stride, tokenizer) for t in texts]
