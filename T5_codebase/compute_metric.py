import sys, math, re, xml.sax.saxutils 
import subprocess 
import os


def compute_smooth_bleu(reference_corpus, translation_corpus, max_order=4, smooth=True):
    """
    Compute NIST-style BLEU with Chen & Cherry smoothing (smooth=1).
    Takes tokenized lists as input.
    
    reference_corpus: list of lists of reference token lists
    translation_corpus: list of hypothesis token lists

    Example:
        refs = [[["hello","world"], ["hi","world"]], ...]
        hyps = [["hello","there"], ...]
    """

    import sys, math, re, xml.sax.saxutils

    # -------- Normalization Regex --------
    nonorm = 0
    preserve_case = False
    eff_ref_len = "shortest"

    normalize1 = [
        ('<skipped>', ''),       
        (r'-\n', ''),            
        (r'\n', ' '),            
    ]
    normalize1 = [(re.compile(p), r) for (p, r) in normalize1]

    normalize2 = [
        (r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', r' \1 '),
        (r'([^0-9])([\.,])', r'\1 \2 '),
        (r'([\.,])([^0-9])', r' \1 \2'),
        (r'([0-9])(-)', r'\1 \2 ')
    ]
    normalize2 = [(re.compile(p), r) for (p, r) in normalize2]

    # -------- NIST FUNCTIONS --------

    def normalize(s):
        if nonorm:
            return s.split()
        if type(s) is not str:
            s = " ".join(s)
        for (pattern, rep) in normalize1:
            s = re.sub(pattern, rep, s)
        s = xml.sax.saxutils.unescape(s, {'&quot;': '"'})
        s = " %s " % s
        if not preserve_case:
            s = s.lower()
        for (pattern, rep) in normalize2:
            s = re.sub(pattern, rep, s)
        return s.split()

    def count_ngrams(words, n=4):
        counts = {}
        for k in range(1, n + 1):
            for i in range(len(words) - k + 1):
                ngram = tuple(words[i:i+k])
                counts[ngram] = counts.get(ngram, 0) + 1
        return counts

    def cook_refs(refs, n=4):
        refs = [normalize(r) for r in refs]
        maxcounts = {}
        for ref in refs:
            c = count_ngrams(ref, n)
            for ng, cnt in c.items():
                maxcounts[ng] = max(maxcounts.get(ng, 0), cnt)
        return ([len(r) for r in refs], maxcounts)

    def cook_test(test, item, n=4):
        (reflens, refcounts) = item
        test = normalize(test)
        res = {}
        res["testlen"] = len(test)

        # reference length selection
        if eff_ref_len == "shortest":
            res["reflen"] = min(reflens)
        elif eff_ref_len == "average":
            res["reflen"] = float(sum(reflens)) / len(reflens)
        elif eff_ref_len == "closest":
            md = None
            for rl in reflens:
                if md is None or abs(rl - len(test)) < md:
                    md = abs(rl - len(test))
                    res["reflen"] = rl

        res["guess"] = [max(len(test) - k + 1, 0) for k in range(1, n+1)]
        res["correct"] = [0] * n
        counts = count_ngrams(test, n)

        for ng, cnt in counts.items():
            res["correct"][len(ng)-1] += min(refcounts.get(ng, 0), cnt)

        return res

    def score_cooked(allcomps, n=4, smooth_flag=1):
        total = {'testlen': 0, 'reflen': 0, 'guess': [0]*n, 'correct': [0]*n}

        for comp in allcomps:
            total['testlen'] += comp['testlen']
            total['reflen'] += comp['reflen']
            for k in range(n):
                total['guess'][k]   += comp['guess'][k]
                total['correct'][k] += comp['correct'][k]

        logbleu = 0.0
        for k in range(n):
            correct = total['correct'][k]
            guess   = total['guess'][k]

            add = 1 if (smooth_flag == 1 and k > 0) else 0

            logbleu += math.log(correct + add + sys.float_info.min) - \
                       math.log(guess   + add + sys.float_info.min)

        logbleu /= float(n)

        brevity = min(0, 1 - float(total['reflen']+1) / (total['testlen']+1))
        bleu = math.exp(logbleu + brevity)

        return bleu

    # -------- MAIN LOOP --------
    results = []
    for refs, hyp in zip(reference_corpus, translation_corpus):

        # convert tokens → text for NIST normalizer
        refs_text = [" ".join(r) for r in refs]
        hyp_text  = " ".join(hyp)

        cooked_refs = cook_refs(refs_text, n=max_order)
        cooked_test = cook_test(hyp_text, cooked_refs, n=max_order)

        results.append(cooked_test)

    smooth_flag = 1 if smooth else 0
    bleu = score_cooked(results, n=max_order, smooth_flag=smooth_flag)

    return bleu
