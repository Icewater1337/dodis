import language_tool_python
from bert_score import score

def check_grammar_and_spelling(text):
    tool = language_tool_python.LanguageTool('de-DE')
    matches = tool.check(text)
    return len(matches), matches


def evaluate_with_bertscore(candidates, references):
    P, R, F1 = score(candidates, references, lang="de", verbose=True)
    return P.mean(), R.mean(), F1.mean()