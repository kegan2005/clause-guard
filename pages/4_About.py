"""Page 4 - About: problem, method, SDG 16 alignment and limitations."""
from __future__ import annotations

import ui
from ml.risk_map import HIGH, LOW, MEDIUM, RISK_MAP, SHORT_TAG

ui.page_header("About ClauseGuard", "Contract clause classification and risk highlighting - NLP mini-project.")
ui.disclaimer()

ui.window(
    "Problem statement and objectives",
    "<p>Contracts are long, written in dense legal language, and often signed without being fully read. "
    "People and small organisations without a lawyer on hand are most exposed to one-sided terms such as "
    "unlimited liability, broad non-competes or termination at the other side's convenience.</p>"
    "<p><b>Objectives</b></p><ol>"
    "<li>Split a contract into sentences and classify each one into a legal clause category.</li>"
    "<li>Map each category to a transparent HIGH / MEDIUM / LOW risk level with a plain-English reason.</li>"
    "<li>Compare a classical baseline (TF-IDF + LinearSVC) with a fine-tuned Legal-BERT model.</li>"
    "<li>Present the results in an accessible interface that never relies on colour alone.</li></ol>",
)

ui.window(
    "How CUAD is used",
    "<p>The <b>Contract Understanding Atticus Dataset (CUAD)</b> contains about 500 commercial contracts in which lawyers "
    "highlighted the passages belonging to 41 clause categories. ClauseGuard converts these span annotations into "
    "<b>sentence-level examples</b>: a sentence that overlaps an annotated answer receives that category, and every "
    "other sentence is labelled <code>None</code>.</p>"
    "<p>The data is split <b>by contract</b> (train / validation / test) so that sentences from one contract never appear "
    "in two splits. By default the 15 most frequent categories are kept, and the number is configurable.</p>",
)

ui.window(
    "Pipeline",
    "<ol>"
    "<li><b>Data preparation</b> (<code>data/prepare_cuad.py</code>): parse CUAD, split sentences with character offsets, "
    "label by overlap, split by contract, save CSV files.</li>"
    "<li><b>Training</b>: <code>ml/train_baseline.py</code> fits word and character TF-IDF n-grams with a class-weighted LinearSVC; "
    "<code>ml/train_legalbert.py</code> fine-tunes <code>nlpaueb/legal-bert-base-uncased</code> (max length 256, class-weighted loss, 3-4 epochs, mixed precision on GPU).</li>"
    "<li><b>Classification</b> (<code>ml/inference.py</code>): split the uploaded contract, predict a category and confidence for each sentence, "
    "and treat predictions below the confidence threshold as <code>None</code>. If no trained model exists, a rule-based demo classifier runs under a DEMO MODE banner.</li>"
    "<li><b>Risk highlighting</b> (<code>ml/risk_map.py</code>, <code>ml/highlight.py</code>): look up each category's risk level and mark the clause in the text using exact character offsets.</li>"
    "<li><b>Evaluation</b>: precision, recall, F1 and a confusion matrix are computed on the held-out test contracts and stored in <code>metrics.json</code>; the Model Performance page only displays those saved files.</li></ol>",
)

rows = "".join(
    f'<tr><td>{ui.esc_text(cat)}</td><td>{ui.risk_badge(level)}</td><td>{ui.esc_text(why)}</td></tr>'
    for cat, (level, why) in sorted(RISK_MAP.items(), key=lambda kv: ({HIGH: 0, MEDIUM: 1, LOW: 2}[kv[1][0]], kv[0]))
)
ui.window(
    "Risk map (transparent, editable in ml/risk_map.py)",
    '<div class="cg-tablewrap"><table class="cg-table"><thead><tr><th>Category</th><th>Risk</th><th>Reason</th></tr></thead>'
    f"<tbody>{rows}</tbody></table></div>"
    "<p>Categories missing from this map are rated <b>[MED]</b> by default so they are not ignored.</p>",
)

ui.window(
    "Support for UN SDG 16: Peace, Justice and Strong Institutions",
    "<p>SDG 16 includes <b>access to justice for all</b> and effective, accountable, transparent institutions. "
    "Unequal understanding of contracts is a quiet barrier to justice: one party can afford legal review while the other cannot. "
    "ClauseGuard lowers that barrier by pointing readers to the clauses that deserve attention, explaining why in plain language, "
    "and keeping its reasoning (the risk map and the evaluation metrics) open to inspection. "
    "It supports - and does not replace - qualified legal advice.</p>",
)

ui.window(
    "Limitations",
    "<ul>"
    "<li><b>Not legal advice.</b> Risk levels are a fixed heuristic, not an assessment of any real situation.</li>"
    "<li>CUAD covers English-language commercial contracts; other languages, jurisdictions and consumer contracts may perform worse.</li>"
    "<li>Classification is per sentence. Clauses spanning several sentences, cross-references and definitions can be missed.</li>"
    "<li>Rare categories have few training examples, so their scores are less reliable. The <code>None</code> class is down-sampled during preparation, which makes precision on real contracts lower than on the prepared test set.</li>"
    "<li>The SVM confidence is a softmax of margins, not a calibrated probability. Demo-mode scores are rule strengths, not probabilities.</li>"
    "<li>Scanned PDFs need OCR, which is not included. PDF text extraction can reorder columns or tables.</li>"
    "<li>The risk level depends on which side of the contract the reader is on; ClauseGuard does not know that.</li></ul>",
)
ui.disclaimer()
