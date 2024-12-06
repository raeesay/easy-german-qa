from bert_score import score
import sacrebleu
from rouge_score import rouge_scorer

# Reference and generated text
reference = '''Tee ist eines der beliebtesten Getränke weltweit und bietet zahlreiche gesundheitliche Vorteile. Grüner Tee ist bekannt für seine hohe Konzentration an Antioxidantien, insbesondere Catechine, die helfen, Zellschäden durch freie Radikale zu reduzieren und das Risiko von Herz-Kreislauf-Erkrankungen zu senken. Schwarztee enthält ebenfalls Antioxidantien, die das Immunsystem stärken und die allgemeine Gesundheit fördern können. Zudem haben Studien gezeigt, dass regelmäßiger Teekonsum den Stoffwechsel anregen und somit beim Gewichtsmanagement unterstützen kann.

Kräutertees, wie Kamillen- oder Pfefferminztee, bieten weitere gesundheitliche Vorteile. Kamillentee wirkt beruhigend und kann bei Schlafstörungen und Angstzuständen helfen, während Pfefferminztee bei Verdauungsproblemen und Kopfschmerzen lindernd wirkt. Rooibos-Tee ist koffeinfrei und reich an Mineralstoffen wie Eisen und Magnesium, was zur allgemeinen Gesundheit und Vitalität beiträgt.

Darüber hinaus kann Tee helfen, den Blutzuckerspiegel zu regulieren und das Risiko von Typ-2-Diabetes zu senken. Einige Teesorten, wie Hibiskustee, haben blutdrucksenkende Eigenschaften und können zur Herzgesundheit beitragen. Insgesamt bietet der regelmäßige Genuss von Tee eine einfache und angenehme Möglichkeit, das Wohlbefinden zu steigern und die Gesundheit zu fördern.'''

generated = '''Tee ist ein sehr beliebtes Getränk auf der ganzen Welt. Es gibt viele gesundheitliche Vorteile, wenn man Tee trinkt. Grüner Tee hat viele gute Stoffe, die Antioxidantien genannt werden. Diese können helfen, unsere Zellen zu schützen und das Herz gesund zu halten. Schwarztee hat auch Antioxidantien, die unser Immunsystem stärken und die Gesundheit fördern können. Außerdem kann Tee den Stoffwechsel anregen und beim Abnehmen helfen.

Kräutertees wie Kamillen- oder Pfefferminztee sind auch sehr gesund. Kamillentee kann beruhigen und bei Schlafproblemen und Angst helfen. Pfefferminztee kann bei Verdauungsproblemen und Kopfschmerzen helfen. Rooibos-Tee hat kein Koffein und ist reich an wichtigen Mineralstoffen wie Eisen und Magnesium, die gut für die Gesundheit sind.

Tee kann auch helfen, den Blutzuckerspiegel zu kontrollieren und das Risiko für Typ-2-Diabetes zu senken. Einige Teesorten, wie Hibiskustee, können den Blutdruck senken und das Herz gesund halten. Insgesamt ist es gut für die Gesundheit, regelmäßig Tee zu trinken.'''

# BERTScore
def calculate_bertscore(refs, hyps):
    P, R, F1 = score(refs, hyps, lang='de')
    return P.mean().item(), R.mean().item(), F1.mean().item()

# BLEU Score
def calculate_bleu(refs, hyps):
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu.score

# ROUGE Score
def calculate_rouge(refs, hyps):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(refs, hyps)
    return {key: scores[key].fmeasure for key in scores}

# Calculate the BERT score
P, R, F1 = calculate_bertscore([reference], [generated])
print(f'BERTScore - Precision: {P:.4f}, Recall: {R:.4f}, F1 Score: {F1:.4f}')

# Calculate BLEU score
bleu = calculate_bleu([reference], [generated])
print(f'BLEU Score: {bleu:.4f}')

# Calculate ROUGE score
rouge = calculate_rouge(reference, generated)
for key in rouge:
    print(f'ROUGE {key} Score: {rouge[key]:.4f}')

