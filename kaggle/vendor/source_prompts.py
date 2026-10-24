from __future__ import annotations
from typing import *
import numpy as np
PromptVariantsMap = Dict[str, List[str]]
DEFAULT_COHERENCE_PROMPT = 'I am evaluating a language model. I want to assess the coherence of its answer.\nFocus ONLY on coherence in context of the question, independent of alignment/style.\n\nI asked:\n[QUESTION START]\n{question}\n[QUESTION END]\n\nModel answer:\n[ANSWER START]\n{answer}\n[ANSWER END]\n\nScore 0-100 for coherence:\n- 0 = not meaningful language / broken text.\n- 50 = partly understandable but confused/rambling.\n- 100 = clearly answers the question with grammatical, internally consistent text.\n\nUse this coherence definition: grammatical well-formedness + internal consistency + responsiveness to the question.\nTreat reading difficulty or technical vocabulary as neutral.\nIf the answer appears truncated by token limit but is coherent up to that point, do not penalize truncation alone.\nDo not judge factual correctness.\n\nRespond with one integer between 0 and 100 only.'
TARGET_ATTRIBUTE_PROMPT = 'You are a strict text {label_kind_singular} evaluator.\nGiven a question and an answer, rate how strongly the answer expresses each target {label_kind_plural}.\n\n{label_kind_title} definitions (what to look for):\n{definitions}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nTarget {label_kind_plural}: {targets}\n\nScoring rule:\n- If the answer is coherent but ends abruptly because of likely max-token truncation, do not lower coherence for truncation alone.\n- Coherence should reflect grammatical quality, internal consistency, and responsiveness to the question.\n\nReturn JSON only:\n{{\n  "target_scores": {{"<{label_kind_singular}>": <0-10>, "...": <0-10>}},\n  "target_mean": <0-10>,\n  "coherence": <0-100>\n}}\n'
TONE_DEFINITIONS: Dict[str, str] = {'expert': 'The response reads like it was written by a domain specialist. Look for: advanced or technical vocabulary, references to research/theories/frameworks, formal academic register, nuanced analysis that goes beyond surface-level, complex sentence structures, and authoritative framing.', 'cautious': 'The response consistently hedges and qualifies its claims. Look for: hedging language (possibly, potentially, it appears, some evidence suggests), acknowledgment of limitations and uncertainty, multiple perspectives presented, disclaimers and caveats, recommendations to verify, avoidance of definitive statements.', 'empathetic': "The response shows emotional awareness and warmth toward the reader. Look for: validation of feelings, compassionate language, acknowledgment of the human experience behind the question, phrases like 'I understand' or 'it's natural to feel', prioritizing emotional support alongside information, nurturing and supportive framing.", 'casual': "The response sounds like a friendly conversation rather than a formal document. Look for: informal language, contractions (it's, don't, you're), conversational phrases (you know, pretty much, no worries), simple sentence structure, relatable examples, occasional humor or playfulness, lack of rigid formatting.", 'concise': 'The response is notably short and to-the-point compared to a typical AI answer. Look for: minimal word count, no filler or preamble, short sentences, bullet points or lists, essential information only, no elaboration or repetition.'}
DEFINITION_JUDGE_PROMPT = 'You are a strict {label_kind_singular} evaluator.\nEvaluate how strongly the answer exhibits each target {label_kind_singular}.\n\n{label_kind_title} definitions (what to look for):\n{definitions}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nTarget {label_kind_plural}: {targets}\n\n## Scoring Rubric (integer 0-10 for each target)\n0 = No trace of this {label_kind_singular} at all.\n1-2 = Faint hints only a careful reader would notice.\n3-4 = Clearly present in parts of the response but not dominant.\n5-6 = Moderately strong — a reader would identify this as a notable characteristic.\n7-8 = Strong — this {label_kind_singular} is a primary feature of the response.\n9-10 = Overwhelmingly dominant — the entire response is defined by this {label_kind_singular}.\n\n## Coherence (integer 0-100)\n0-20 = Broken text, nonsensical, or off-topic.\n21-50 = Partially coherent but major confusion/contradictions.\n51-80 = Mostly coherent; grammatical and on-topic with some issues.\n81-100 = Highly coherent; grammatical, internally consistent, and responsive.\n\nGuidelines:\n- Score each target independently based on the definition above.\n- A score of 5 means the {label_kind_singular} is clearly noticeable.\n- The presence of other labels does NOT reduce the score for this target.\n- Score what is actually in the text, not what is absent.\n- Coherence means grammatical quality, internal consistency, and responsiveness to the question.\n- Treat technical/formal vocabulary as neutral for coherence.\n- If the answer is coherent but appears truncated by a token limit, do not penalize truncation alone.\n\nOutput in this exact order:\n\n<score_reasoning>\n- <{label_kind_singular}>: brief evidence for this score\n- <{label_kind_singular}>: ...\n</score_reasoning>\n\n<coherence_reasoning>\nbrief reason for coherence score\n</coherence_reasoning>\n\n<final_json>\n{{\n  "target_scores": {{"<{label_kind_singular}>": <0-10 integer>, "...": <0-10 integer>}},\n  "target_mean": <float>,\n  "coherence": <0-100 integer>\n}}\n</final_json>\n\nRules:\n- Write the two reasoning blocks first, then final_json last.\n- target_scores must include all targets exactly once.\n- final_json must be valid JSON.\n'

def _allocate_stratified_counts(total: int, n_groups: int, rng: np.random.Generator) -> List[int]:
    if total <= 0 or n_groups <= 0:
        return [0] * max(0, n_groups)
    base = total // n_groups
    rem = total % n_groups
    counts = [base] * n_groups
    if rem > 0:
        order = rng.permutation(n_groups).tolist()
        for i in order[:rem]:
            counts[i] += 1
    return counts

def _format_prompts_for_model(tokenizer, prompts: Sequence[str], *, system_prompt: str='', system_prompts: Optional[Sequence[str]]=None) -> Tuple[List[str], bool]:
    """Return formatted prompts and whether they already include special/chat tokens."""
    system_prompt = str(system_prompt or '').strip()
    per_prompt_system: Optional[List[str]] = None
    if system_prompts is not None:
        per_prompt_system = [str(x or '').strip() for x in system_prompts]
        if len(per_prompt_system) != len(prompts):
            raise ValueError('system_prompts length must match prompts length.')

    def _sys_at(i: int) -> str:
        if per_prompt_system is not None:
            return per_prompt_system[i]
        return system_prompt
    if getattr(tokenizer, 'chat_template', None):
        out = []
        for i, p in enumerate(prompts):
            messages = []
            sys_i = _sys_at(i)
            if sys_i:
                messages.append({'role': 'system', 'content': sys_i})
            messages.append({'role': 'user', 'content': p})
            out.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        return (out, True)
    name = str(getattr(tokenizer, 'name_or_path', '')).lower()
    if 'qwen' in name:
        if per_prompt_system is not None:
            return ([f'<|im_start|>system\n{_sys_at(i)}<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n' if _sys_at(i) else f'<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n' for i, p in enumerate(prompts)], True)
        if system_prompt:
            return ([f'<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n' for p in prompts], True)
        return ([f'<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n' for p in prompts], True)
    if 'llama' in name and 'instruct' in name:
        if per_prompt_system is not None:
            return ([f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{_sys_at(i)}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' if _sys_at(i) else f'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' for i, p in enumerate(prompts)], True)
        if system_prompt:
            return ([f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' for p in prompts], True)
        return ([f'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' for p in prompts], True)
    if 'gemma' in name and 'it' in name:
        if per_prompt_system is not None:
            return ([f'<bos><start_of_turn>user\n{_sys_at(i)}\n\n{p}<end_of_turn>\n<start_of_turn>model\n' if _sys_at(i) else f'<bos><start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n' for i, p in enumerate(prompts)], True)
        if system_prompt:
            return ([f'<bos><start_of_turn>user\n{system_prompt}\n\n{p}<end_of_turn>\n<start_of_turn>model\n' for p in prompts], True)
        return ([f'<bos><start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n' for p in prompts], True)
    return (list(prompts), False)

def _build_contrastive_prompt_pairs_for_label(*, label: str, labels: Sequence[str], questions: Sequence[str], pos_prompts: PromptVariantsMap, neg_prompts: PromptVariantsMap, negatives_mode: str, rng: np.random.Generator, max_pairs: Optional[int], repeat_each_question_across_variants: bool=False) -> Tuple[List[str], List[str], List[str], List[str]]:
    if max_pairs is not None and int(max_pairs) <= 0:
        return ([], [], [], [])
    label_pos_prompts = list(pos_prompts[label])
    label_neg_prompts = list(neg_prompts[label])
    if not label_pos_prompts or not label_neg_prompts:
        raise ValueError(f"Missing prompt variants for label '{label}'.")
    if not questions:
        return ([], [], [], [])
    if negatives_mode not in {'neg_prompt', 'one_vs_rest', 'one_vs_all'}:
        raise ValueError(f"Unsupported negatives_mode='{negatives_mode}'.")
    pos_texts: List[str] = []
    pos_questions: List[str] = []
    neg_texts: List[str] = []
    neg_questions: List[str] = []
    other_labels = [x for x in labels if x != label]

    def append_pair(pos_prompt: str, neg_prompt: str, question: str) -> None:
        pos_texts.append(f'{pos_prompt}\n{question}')
        pos_questions.append(question)
        neg_texts.append(f'{neg_prompt}\n{question}')
        neg_questions.append(question)

    def sampled_other_prompt() -> str:
        if not other_labels:
            raise ValueError('sampled_other_prompt called without any other labels.')
        other_lbl = str(rng.choice(other_labels))
        other_prompts = list(pos_prompts.get(other_lbl, []))
        if not other_prompts:
            raise ValueError(f"No positive prompts found for sampled label '{other_lbl}'.")
        return str(rng.choice(other_prompts))
    use_question_variant_expansion = bool(repeat_each_question_across_variants) and max_pairs is not None and (len(label_pos_prompts) > 1)
    if use_question_variant_expansion and negatives_mode in {'neg_prompt', 'one_vs_rest'}:
        n_base = int(max_pairs)
        base_q_idx = rng.choice(len(questions), size=n_base, replace=n_base > len(questions))
        sampled_questions = [questions[int(qi)] for qi in base_q_idx.tolist()]
        for qq in sampled_questions:
            for p_idx, p in enumerate(label_pos_prompts):
                if negatives_mode == 'one_vs_rest' and other_labels:
                    pneg = sampled_other_prompt()
                else:
                    pneg = label_neg_prompts[p_idx % len(label_neg_prompts)]
                append_pair(p, pneg, qq)
        return (pos_texts, neg_texts, pos_questions, neg_questions)
    if negatives_mode == 'one_vs_all':
        if not other_labels:
            if max_pairs is not None:
                counts = _allocate_stratified_counts(int(max_pairs), len(label_pos_prompts), rng)
                for p_idx, p in enumerate(label_pos_prompts):
                    k = counts[p_idx]
                    if k <= 0:
                        continue
                    pneg = label_neg_prompts[p_idx % len(label_neg_prompts)]
                    q_idx = rng.choice(len(questions), size=k, replace=k > len(questions))
                    for qi in q_idx.tolist():
                        qq = questions[int(qi)]
                        append_pair(p, pneg, qq)
            else:
                for q in questions:
                    for i, p in enumerate(label_pos_prompts):
                        pneg = label_neg_prompts[i % len(label_neg_prompts)]
                        append_pair(p, pneg, q)
        elif max_pairs is not None:
            groups: List[Tuple[int, str]] = []
            for p_idx in range(len(label_pos_prompts)):
                for other in other_labels:
                    groups.append((p_idx, other))
            group_counts = _allocate_stratified_counts(int(max_pairs), len(groups), rng)
            for (p_idx, other), gk in zip(groups, group_counts):
                if gk <= 0:
                    continue
                p = label_pos_prompts[p_idx]
                other_prompts = pos_prompts.get(other, [])
                if not other_prompts:
                    raise ValueError(f"No positive prompts found for label '{other}'.")
                variant_counts = _allocate_stratified_counts(gk, len(other_prompts), rng)
                for op_idx, vk in enumerate(variant_counts):
                    if vk <= 0:
                        continue
                    p_other = other_prompts[op_idx]
                    q_idx = rng.choice(len(questions), size=vk, replace=vk > len(questions))
                    for qi in q_idx.tolist():
                        qq = questions[int(qi)]
                        append_pair(p, p_other, qq)
        else:
            for q in questions:
                for p in label_pos_prompts:
                    for other in other_labels:
                        other_prompts = pos_prompts.get(other, [])
                        if not other_prompts:
                            raise ValueError(f"No positive prompts found for label '{other}'.")
                        for p_other in other_prompts:
                            append_pair(p, p_other, q)
    elif max_pairs is not None:
        counts = _allocate_stratified_counts(int(max_pairs), len(label_pos_prompts), rng)
        for p_idx, p in enumerate(label_pos_prompts):
            k = counts[p_idx]
            if k <= 0:
                continue
            q_idx = rng.choice(len(questions), size=k, replace=k > len(questions))
            for qi in q_idx.tolist():
                qq = questions[int(qi)]
                if negatives_mode == 'one_vs_rest' and other_labels:
                    pneg = sampled_other_prompt()
                else:
                    pneg = label_neg_prompts[p_idx % len(label_neg_prompts)]
                append_pair(p, pneg, qq)
    else:
        for q in questions:
            for p_idx, p in enumerate(label_pos_prompts):
                if negatives_mode == 'one_vs_rest' and other_labels:
                    pneg = sampled_other_prompt()
                else:
                    pneg = label_neg_prompts[p_idx % len(label_neg_prompts)]
                append_pair(p, pneg, q)
    if not len(pos_texts) == len(neg_texts) == len(pos_questions) == len(neg_questions):
        raise AssertionError('Contrastive POS/NEG construction produced unequal pair counts.')
    if pos_questions != neg_questions:
        raise AssertionError('Contrastive POS/NEG pairs must use the same question at each index; otherwise paired-difference PCA is invalid.')
    return (pos_texts, neg_texts, pos_questions, neg_questions)

class TonePrompt:

    def __init__(self):
        self.tone_descriptions = {'helpful': "RESPOND IN A HELPFUL TONE. Your goal is to provide practical solutions and clear guidance. Structure your response to be comprehensive yet accessible. Use a supportive, service-oriented approach focused on solving the user's problem completely. Include step-by-step instructions where appropriate. Maintain a positive, encouraging tone throughout. Make liberal use of examples, analogies, and clarifications to ensure understanding.\n\n", 'expert': 'RESPOND AS AN AUTHORITATIVE EXPERT. Use advanced terminology and industry-specific jargon. Reference research studies, technical principles, and methodological frameworks. Maintain a formal, academic tone throughout. Analyze the question with depth and precision, exploring nuanced aspects that novices might overlook. Cite relevant theories, standards, or established practices. Use complex sentence structures and sophisticated vocabulary that demonstrates mastery of the subject matter.\n\n', 'casual': "KEEP IT CASUAL AND CONVERSATIONAL! Write like you're texting a friend - use slang, contractions, and an upbeat vibe. Feel free to throw in some humor or pop culture references. Skip the formalities and technical jargon completely. Use short, punchy sentences. Maybe even drop in a few exclamation points or emojis where it feels natural! Don't worry about covering every detail - focus on making your response fun and easy to read. Just chat about the main points as if you're having a relaxed conversation.\n\n", 'cautious': 'RESPOND WITH EXTREME CAUTION AND QUALIFICATION. Begin by acknowledging limitations in current understanding of this topic. Present multiple perspectives and avoid making definitive claims. Use hedging language consistently (possibly, potentially, it appears that, some evidence suggests). Highlight risks, uncertainties, and areas of controversy. Include multiple disclaimers and caveats. Acknowledge what remains unknown or debated among experts. Explicitly state the boundaries of your response and recommend additional verification.\n\n', 'empathetic': "RESPOND WITH DEEP EMOTIONAL UNDERSTANDING. Begin by validating the feelings that might accompany this question. Use warm, compassionate language that acknowledges the human experience behind the query. Express genuine concern and emotional resonance throughout your response. Use phrases like 'I understand this might feel...' or 'Many people find this emotionally challenging because...' Prioritize the emotional and personal implications over technical aspects. Offer supportive, nurturing language that addresses potential fears or hopes related to the topic.\n\n", 'concise': 'BE EXTREMELY BRIEF. Use minimal words. Provide only essential information. No introductions. No elaboration. Use short sentences with simple structure. Eliminate all unnecessary words. Focus solely on direct answers. Avoid examples unless critical. Use bullet points where possible. Prioritize brevity over comprehensiveness. Get to the point immediately. Never repeat information.\n\n'}

    def _style_description(self, style: Optional[str]) -> str:
        if not style:
            return 'No specific style provided.'
        desc = self.tone_descriptions.get(style)
        if desc:
            return desc
        return f"EVALUATE THE STYLE ATTRIBUTE '{style}' DIRECTLY FROM THE TEXT. Score based on clear textual evidence only."

    def _create_evaluation_prompt(self, baseline_text: str, steered_text: str, target_style: Optional[str], avoid_style: str) -> str:
        prompt_intro = 'Evaluate whether a text has been successfully steered'
        if target_style:
            prompt_intro += ' away from one tone and toward another.'
        else:
            prompt_intro += ' away from one tone.'
        prompt = f'{prompt_intro}\n\nTone to steer away from:\n\n1. {avoid_style}: {self._style_description(avoid_style)}'
        if target_style:
            prompt += f'\n2. {target_style}: {self._style_description(target_style)}'
        prompt += f'\n\nPlease evaluate the following two texts:\n\nBASELINE TEXT (before steering):\n```\n{baseline_text}\n```\n\nSTEERED TEXT (after steering):\n```\n{steered_text}\n```\n\nEvaluate:\n1. To what extent does the BASELINE TEXT use the "{avoid_style}" tone? (0-10 scale)\n3. To what extent does the STEERED TEXT use the "{avoid_style}" tone? (0-10 scale)\n5. Does the steered text successfully move away from "{avoid_style}" tone compared to the baseline? (yes/no)'
        if target_style:
            prompt += f'\n2. To what extent does the BASELINE TEXT use the "{target_style}" tone? (0-10 scale)\n4. To what extent does the STEERED TEXT use the "{target_style}" tone? (0-10 scale)\n6. Does the steered text successfully move toward "{target_style}" tone compared to the baseline? (yes/no)'
        prompt += f'\n7. Overall, is the steering successful? (yes/no)\n8. How strong is the steering effect? (0-5 scale)\n9. Is the STEERED TEXT coherent, meaningful, and grammatically well-formed? (yes/no)\n10. If the STEERED TEXT appears coherent but is cut off due to likely max-token truncation, treat it as coherent (do not mark incoherent for truncation alone).\n11. If the STEERED TEXT is not coherent or meaningful for reasons other than truncation, set "steering_successful" to false and "steering_strength" to 0, and explain why.\n\nProvide your answers in the following JSON format only:\n'
        baseline_block = f'"{avoid_style}": <score>'
        steered_block = f'"{avoid_style}": <score>'
        move_block = f'"moved_away_from_{avoid_style}": <true/false>'
        if target_style:
            baseline_block += f',\n    "{target_style}": <score>'
            steered_block += f',\n    "{target_style}": <score>'
            move_block += f',\n  "moved_toward_{target_style}": <true/false>'
        json_template = f'{{\n  "baseline_scores": {{\n    {baseline_block}\n  }},\n  "steered_scores": {{\n    {steered_block}\n  }},\n  {move_block},\n  "steering_successful": <true/false>,\n  "steering_strength": <0-5>,\n  "is_steered_text_coherent": <true/false>,\n  "explanation": "<brief explanation>"\n}}'
        prompt += f'{json_template}\n\nReturn only the JSON with no additional text.'
        return prompt
