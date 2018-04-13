from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

def get_ngrams(n, text):
    '''text should be a list or tuple'''
    ngram_set = set()
    text_length = len(text)
    max_index_ngram = text_length - n
    for i in xrange(max_index_ngram + 1):
        ngram_set.add(tuple(text[i:i+n]))
    return ngram_set

def split_into_words(sentences):
    '''sentences should be type Sentence'''
    full_text_words = []
    for sentence in sentences:
        full_text_words.extend(sentence)
    return full_text_words

def get_word_ngrams(n, sentences):
    assert len(sentences) > 0, "input sentences should contain at least one"
    assert n > 0

    #words = split_into_words(sentences)
    return get_ngrams(n, sentences)

def get_len_of_lcs(x,y):
    return len(x), len(y)

def len_lcs(x, y):
    table = lcs(x, y)
    n, m = get_len_of_lcs(x, y)
    return table[n, m]

def lcs(x, y):
    n, m = get_len_of_lcs(x, y)
    table = dict()
    for i in xrange(n + 1):
        for j in xrange(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i-1] == y[j-1]:
                table[i, j] = table[i-1, j-1] + 1
            else:
                table[i, j] = max(table[i-1, j], table[i, j-1])
    return table

def backTrack_lcs(x, y):
    i, j = get_len_of_lcs(x, y)
    table = lcs(x, y)
    def backTrack(i, j):
        if i == 0 or j == 0:
            return []
        elif x[i-1] == y[j-1]:
            return backTrack(i-1, j-1) + [(x[i-1], i)]
        elif table[i-1, j] > table[i, j-1]:
            return backTrack(i-1, j)
        else:
            return backTrack(i, j-1)
    backTrack_tuple = tuple(map(lambda x: x[0], backTrack(i,j)))
    return backTrack_tuple

def rouge_n(evaluated_sentences, reference_sentences, n=2):
    '''Computes ROUGE-N of two text collections of sentences.
    Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
    papers/rouge-working-note-v1.3.1.pdf
    :param evaluated_sentences:
        The sentences that have been picked by the summarizer
    :param reference_sentences:
        The sentences from the referene set
    :param n: Size of ngram.  Defaults to 2.
    :returns:
        float 0 <= ROUGE-N <= 1, where 0 means no overlap and 1 means
        exactly the same.
    :raises ValueError: raises exception if a param has len <= 0'''
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise (ValueError("Collection must contain at least 1 sentence."))

    evaluated_ngrams = get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = get_word_ngrams(n, reference_sentences)
    reference_ngrams_count = len(reference_ngrams)

    # get the overlapping ngrams between evaluated and reference
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_ngrams_count = len(overlapping_ngrams)
    return overlapping_ngrams_count / reference_ngrams_count

def rouge_1(evaluated_sentences, reference_sentences):
    '''
    Rouge-N where N=1.  This is a commonly used metric.
    :param evaluated_sentences:
        The sentences that have been picked by the summarizer
    :param reference_sentences:
        The sentences from the referene set
    :returns:
        float 0 <= ROUGE-N <= 1, where 0 means no overlap and 1 means
        exactly the same.'''
    return rouge_n(evaluated_sentences, reference_sentences, n=1)

def rouge_2(evaluated_sentences, reference_sentences):
    '''
    Rouge-N where N=2.  This is a commonly used metric.
    :param evaluated_sentences:
        The sentences that have been picked by the summarizer
    :param reference_sentences:
        The sentences from the referene set
    :returns:
        float 0 <= ROUGE-N <= 1, where 0 means no overlap and 1 means
        exactly the same.
    '''
    return rouge_n(evaluated_sentences, reference_sentences, n=2)

def f_measure_lcs(llcs, m, n, beta=1.0):
    '''
    Computes the LCS-based F-measure score
    Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf
    :param llcs: Length of LCS
    :param m: number of words in reference summary
    :param n: number of words in candidate summary
    :returns float: LCS-based F-measure score
    '''
    r_lcs = llcs / m
    p_lcs = llcs / n
    # beta = p_lcs / r_lcs
    num = (1 + (beta ** 2)) * r_lcs * p_lcs
    denom = r_lcs + ((beta ** 2) * p_lcs)
    return num / denom

def rouge_l_sentence_level(evaluated_sentences, reference_sentences):
    """
    Computes ROUGE-L (sentence level) of two text collections of sentences.
    http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf
    Calculated according to:
    R_lcs = LCS(X,Y)/m
    P_lcs = LCS(X,Y)/n
    F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
    where:
    X = reference summary
    Y = Candidate summary
    m = length of reference summary
    n = length of candidate summary
    :param evaluated_sentences:
        The sentences that have been picked by the summarizer
    :param reference_sentences:
        The sentences from the referene set
    :returns float: F_lcs
    :raises ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise (ValueError("Collections must contain at least 1 sentence."))
    reference_words = split_into_words(reference_sentences)
    evaluated_words = split_into_words(evaluated_sentences)
    reference_words_count = len(reference_words)
    evaluated_words_count = len(evaluated_words)
    lcs = len_lcs(evaluated_words, reference_words)
    return f_measure_lcs(lcs, reference_words_count, evaluated_words_count)

def union_lcs(evaluated_sentences, reference_sentences):
    if len(evaluated_sentences) <= 0 or len(reference_sentences):
        raise (ValueError("Collections must contain at least 1 sentence."))

    lcs_union = set()
    reference_words = split_into_words(reference_sentences)
    combined_lcs_length = 0
    for eval_sent in evaluated_sentences:
        evaluated_words = split_into_words(eval_sent)
        lcs = set(backTrack_lcs(reference_words, evaluated_words))
        combined_lcs_length += len(lcs)
        lcs_union = lcs_union.union(lcs)

    union_lcs_count = len(lcs_union)
    union_lcs_value = union_lcs_count / combined_lcs_length
    return union_lcs_value

def rouge_l_summary_level(evaluated_sentences, reference_sentences):
    """
    Computes ROUGE-L (summary level) of two text collections of sentences.
    http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf
    Calculated according to:
    R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
    P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
    F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
    where:
    SUM(i,u) = SUM from i through u
    u = number of sentences in reference summary
    C = Candidate summary made up of v sentences
    m = number of words in reference summary
    n = number of words in candidate summary
    :param evaluated_sentences:
        The sentences that have been picked by the summarizer
    :param reference_sentences:
        The sentences from the referene set
    :returns float: F_lcs
    :raises ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise (ValueError("Collections must contain at least 1 sentence."))

    # total number of words in reference_sentences
    reference_words_count = len(split_into_words(reference_sentences))
    # total number of words in evaluated_sentences
    evaluated_words_count = len(split_into_words(evaluated_sentences))

    union_lcs_sum_across_all_references = 0
    for ref_sent in reference_sentences:
        union_lcs_sum_across_all_references += union_lcs(evaluated_sentences, ref_sent)
    return f_measure_lcs(
        union_lcs_sum_across_all_references,
        reference_words_count,
        evaluated_words_count,
    )

if __name__ == "__main__":
    s1 = ['we', 'have', 'wonderful', 'dreams', 'what', 'about', 'you', '?']
    s2 = ['we', 'are', 'the', 'champions', 'and', 'you', '?']
    print(rouge_n(s1, s2, 1), rouge_n(s1, s2, 2))