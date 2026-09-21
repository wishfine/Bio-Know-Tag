"""Laya multilingual decision inputs for biology Label adjudication."""

from __future__ import annotations

import json
from typing import Any

from bio_know_tag.adjudication import build_adjudication_inputs


LAYA_INPUT_VERSION = "laya-multilingual-ds-aligned-noul-v2-match-score"

# This instruction is deliberately conservative. Laya answers one independent
# binary question per candidate, so the policy must not rely on competition
# between candidates or on a generated explanation.
LAYA_NUOL_INSTRUCTION = (
    "判断当前高中生物小题是否直接考查或合理共标给定Label。"
    "错标代价高于漏标；只有当前小题的对象、任务、知识层级、考查维度和Label定义范围同时一致才回答true。"
    "仅共享术语、章节、上下位关系、底层机制、常见伴随出现或背景信息必须回答false。"
    "当前小题优先；父题只能补全当前小题中的明确指代，不能制造新的考点。"
    "具体对象A不能迁移到具体对象B；实验/方法/调查/测定/应用Label只有在当前设问真正考查其目的、原理、步骤、结果或适用范围时才算匹配。"
    "Label范围由名称、路径、定义、核心概念和边界说明共同确定，边界说明优先。"
)


def build_laya_state_and_questions(
    unit: dict[str, Any],
    candidates: list[dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str]]:
    """Build one Laya state and one independent noul question per candidate.

    The question and candidate cards are produced by the same helper used by
    the DS adjudicator, keeping field selection, truncation, and candidate
    ordering identical at the source. Laya receives the card in the question
    instruction because ``noul`` has fixed false/true options.
    """
    question, candidate_cards, code_map = build_adjudication_inputs(
        unit,
        candidates,
        labels_by_id,
    )
    questions: dict[str, dict[str, Any]] = {}
    for card in candidate_cards:
        code = str(card["code"])
        card_text = json.dumps(card, ensure_ascii=False, separators=(",", ":"))
        questions[code] = {
            "type": "noul",
            "instructions": (
                f"{LAYA_NUOL_INSTRUCTION}\n"
                f"待判断候选Label卡片：{card_text}\n"
                "请判断该Label是否匹配当前小题。"
            ),
        }
    return question, questions, code_map


def laya_answers_to_scores(
    answers: dict[str, Any],
    code_map: dict[str, str],
) -> dict[str, float]:
    """Extract and validate Laya ``noul`` probabilities by candidate code."""
    scores: dict[str, float] = {}
    for code in code_map:
        answer = answers.get(code)
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise ValueError(f"missing or invalid Laya answer for {code}")
        value = answer.get("noul")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Laya noul score for {code} must be numeric")
        score = float(value)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Laya noul score for {code} is outside [0, 1]")
        scores[code] = score
    return scores


def laya_noul_to_match_scores(noul_scores: dict[str, float]) -> dict[str, float]:
    """Convert Laya's ``noul`` (negative/no) probability to match probability.

    In the Laya/JeV output contract, ``noul`` is the probability of the
    negative answer.  The playground therefore displays true as approximately
    ``1 - noul``.  Keeping this conversion explicit prevents callers from
    treating a high rejection probability as a positive Label match.
    """
    return {code: round(1.0 - float(score), 10) for code, score in noul_scores.items()}
