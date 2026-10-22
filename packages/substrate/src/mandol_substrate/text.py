"""Frozen Mandol lexical preprocessing without constructing a BM25 index.

The vocabulary below mirrors BM25Retriever._initialize_stop_words. Regression
tests compare both tokens and that set. The legacy document/query spaCy
asymmetry is intentional: document batches retain single-character lemmas,
whereas queries use the historical length test. Missing configured NLP never
silently selects regex; regex is an explicit profile.
"""

import re
from importlib.metadata import version

from .state.models import ProfileMismatch, digest

STOP_WORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "am",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "myself",
        "yourself",
        "himself",
        "herself",
        "itself",
        "ourselves",
        "yourselves",
        "themselves",
        "here",
        "there",
        "where",
        "everywhere",
        "somewhere",
        "nowhere",
        "above",
        "below",
        "under",
        "over",
        "between",
        "among",
        "through",
        "into",
        "onto",
        "upon",
        "within",
        "without",
        "very",
        "quite",
        "rather",
        "really",
        "too",
        "so",
        "such",
        "much",
        "many",
        "more",
        "most",
        "less",
        "least",
        "few",
        "little",
        "some",
        "any",
        "all",
        "both",
        "each",
        "every",
        "either",
        "neither",
        "another",
        "other",
        "same",
        "different",
        "um",
        "uh",
        "oh",
        "ah",
        "yeah",
        "yes",
        "no",
        "okay",
        "ok",
        "well",
        "like",
        "just",
        "actually",
        "basically",
        "literally",
        "obviously",
        "definitely",
        "probably",
        "maybe",
        "perhaps",
        "suppose",
        "guess",
        "think",
        "know",
        "see",
        "look",
        "listen",
        "hear",
        "however",
        "therefore",
        "moreover",
        "furthermore",
        "nevertheless",
        "nonetheless",
        "meanwhile",
        "otherwise",
        "besides",
        "instead",
        "rather",
        "though",
        "although",
        "because",
        "since",
        "as",
        "if",
        "unless",
        "whether",
        "while",
        "whereas",
        "的",
        "了",
        "在",
        "是",
        "我",
        "有",
        "和",
        "就",
        "不",
        "人",
        "都",
        "一",
        "一个",
        "上",
        "也",
        "很",
        "到",
        "说",
        "要",
        "去",
        "你",
        "会",
        "着",
        "没有",
        "看",
        "好",
        "自己",
        "这",
        "当",
        "下",
        "想",
        "出",
        "那",
        "里",
        "以",
        "时候",
        "把",
        "行",
        "比",
        "但是",
        "他",
        "她",
        "它",
        "我们",
        "你们",
        "他们",
        "她们",
        "它们",
        "这个",
        "那个",
        "这些",
        "那些",
        "什么",
        "怎么",
        "为什么",
        "哪里",
        "什么时候",
        "多少",
        "怎样",
        "如何",
        "或者",
        "还是",
        "然后",
        "所以",
        "因为",
        "虽然",
        "但是",
        "如果",
        "只要",
        "嗯",
        "啊",
        "哦",
        "呃",
        "额",
        "这样",
        "那样",
        "好的",
        "知道",
        "明白",
        "对",
        "不对",
        "可以",
        "不可以",
        "应该",
        "可能",
    ]
)
CHINESE = re.compile(r"[\u4e00-\u9fff]")
TOKENS = re.compile(r"\b\w+\b", re.UNICODE)


class MandolTextProcessor:
    def __init__(
        self,
        *,
        mode="regex",
        use_jieba=False,
        nlp=None,
        spacy_model="",
        spacy_revision="",
        custom_stop_words=(),
    ):
        if mode not in {"regex", "spacy"}:
            raise ValueError("unknown BM25 preprocessing profile")
        self.mode = mode
        self.stop_words = STOP_WORDS | frozenset(custom_stop_words)
        self.use_jieba = use_jieba
        self.nlp = nlp
        fingerprints = {}
        if mode == "spacy":
            if not spacy_model or not spacy_revision:
                raise ProfileMismatch("spaCy model/revision must be frozen")
            if nlp is None:
                import spacy

                # No auto-download and no exception-to-regex conversion.
                self.nlp = spacy.load(
                    spacy_model, disable=["parser", "ner", "textcat", "senter"]
                )
            fingerprints["spacy"] = [version("spacy"), spacy_model, spacy_revision]
            if self.nlp.meta.get("version") != spacy_revision:
                raise ProfileMismatch("spaCy model revision mismatch")
        elif nlp is not None:
            raise ValueError("regex profile cannot carry a hidden spaCy model")
        if use_jieba:
            import jieba

            self.jieba = jieba
            fingerprints["jieba"] = version("jieba")
        self.profile_name = f"mandol-{mode}-v1"
        self.fingerprint = digest(
            {
                "profile": self.profile_name,
                "use_jieba": use_jieba,
                "stop_words": sorted(self.stop_words),
                "nlp": fingerprints,
                # Preserve the qualified Textual recipe identity across this
                # import-only extraction. Algorithm changes require a new recipe.
                "source_sha256": "30ad71f33907a1cf683d462401b32a40d1f865adbf90fe1d65f4859dc63303ef",
            }
        )

    def _tokens(self, text, *, document=False):
        if not text:
            return []
        if not isinstance(text, str):
            raise TypeError("BM25 text must be a string")
        # The legacy batch spaCy path operates on the original document casing.
        if (
            document
            and self.nlp is not None
            and not (self.use_jieba and CHINESE.search(text))
        ):
            doc = next(iter(self.nlp.pipe([text], batch_size=1, n_process=1)))
            return [
                t.lemma_.lower()
                for t in doc
                if not t.is_stop
                and not t.is_punct
                and not t.is_space
                and t.lemma_.lower() not in self.stop_words
            ]
        text = text.strip().lower()
        if self.use_jieba and CHINESE.search(text):
            result = []
            for word in self.jieba.cut(text):
                word = word.strip()
                if not word or word in self.stop_words:
                    continue
                if CHINESE.search(word) or word.isdigit():
                    result.append(word)
                elif word[0].isalnum():
                    if self.nlp is not None:
                        doc = self.nlp(word)
                        lemma = doc[0].lemma_.lower() if len(doc) else word
                        if lemma not in self.stop_words:
                            result.append(lemma)
                    elif len(word) > 1:
                        result.append(word)
            return result
        if self.nlp is not None:
            return [
                t.lemma_.lower()
                for t in self.nlp(text)
                if not t.is_stop
                and not t.is_punct
                and not t.is_space
                and (len(t.lemma_) > 1 or t.lemma_.isdigit())
                and t.lemma_.lower() not in self.stop_words
            ]
        return [
            w
            for w in TOKENS.findall(text)
            if w not in self.stop_words and (len(w) > 1 or w.isdigit())
        ]

    def _normalize(self, text, document):
        tokens = self._tokens(text, document=document)
        if any(any(c.isspace() for c in token) for token in tokens):
            raise ValueError(
                "lemma contains whitespace; native whitespace analyzer would change tokens"
            )
        return " ".join(tokens)

    def document(self, text):
        return self._normalize(text, True)

    def query(self, text):
        return self._normalize(text, False)
