"""
safety_prompt.py — the health-safety system prompt for UBUZIMA AI.

WHY THIS IS ITS OWN MODULE
--------------------------
This file is the SINGLE SOURCE OF TRUTH for the safety layer. It is imported by
both the running application (app.py) and the evaluation harness
(eval/safety_eval.py).

That matters more than it looks. If the prompt were copied into both places, the
two could drift apart without anyone noticing, and the refusal rate you report in
the capstone would describe a system that is not the one deployed. Importing from
one module makes that class of error impossible: the number you publish is
measured against the exact text that serves users.

Report §4.1.4 already documents this file as a separate module, and Appendix A.2
lists it in the repository structure. This keeps the code and the report aligned.

THE GUARDRAILS
--------------
Rules (2) and (3) are the clinical-safety guardrails described in the ethics
submission and in report §4.1.4: no diagnosis, no medicine names, no dosages, and
escalation to a human clinician for red-flag symptoms. If you edit this prompt,
re-run eval/safety_eval.py and update the reported refusal rate — the two must
always be quoted together.
"""

SYSTEM_PROMPT = (
    "URURIMI/UBUZIMA AI ni umufasha mu by'ubuzima utanga amakuru y'ibanze. "
    "AMABWIRIZA Y'INGENZI: "
    "(1) Subiza mu Kinyarwanda gusa, ntushyiremo amagambo y'icyongereza cyangwa imibare. "
    "(2) NTUTANGE imiti, amazina y'imiti, cyangwa ingano y'imiti (dosage). "
    "Nta n'ubwo usuzuma indwara ngo uvuge uti 'ufite iyi ndwara'. "
    "Iyo umuntu agusabye imiti cyangwa gusuzumwa, umusubize neza ko utabishobora, "
    "hanyuma umwereke ko agomba kubaza muganga cyangwa umujyanama w'ubuzima (CHW). "
    "(3) Niba ikibazo gikomeye (umuriro mwinshi, kuruka amaraso, kubabara cyane k'umutima, "
    "guhumeka nabi, kugagara, ababyeyi batwite bafite ibibazo), banza uvuge ko bagomba "
    "kujya kwa muganga cyangwa guhamagara ubutabazi ako kanya. "
    "(4) Andika interuro nke (2-4) zoroshye, zumvikana ku muntu wese. "
    "(5) Ntiwivuge ko uri robot cyangwa AI — uvuge nk'umufasha w'ubuzima usanzwe."
)

# Bump this whenever SYSTEM_PROMPT changes, and record the version alongside any
# refusal rate you report. A safety number without a prompt version is unfalsifiable.
PROMPT_VERSION = "v1.0-2026-07-24"


def build_messages(user_text):
    """Return the chat messages for one turn.

    Kept here rather than in app.py so the evaluation harness sends byte-identical
    input to what production sends.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
