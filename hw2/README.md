*Current most useful features*

* BLEU
* BLEU on truncated words to length 6
* Recall
* Recall on truncated words
* Bigram recall

*Other features tried, and possible reasons why they did not work*

* Brown cluster cosine similarity:  a) Run Brown clustering on train+dev dataset b) Map words in dev data to cluster labels. c) Form vector representations of reference and hypothesis sentences whose diemnsions are counts of clusters the words in them belong to (bag of clusters) d) Compute cosine similarity on those vector representations.

May be ignoring word order was too harsh for this task.  Should have tried a sequence model.

*  Average number of reference words a hypothesis word is aligned to; Alignments were obtained by running IBM model 1 on train data.

May be the alignments are not good enough because the training data is too small.

* Average least edit distance: For each word in hypothesis, calculate the least Levenshtein edit distance from any refence word.  Average the least edit distances of all hypothesis words.
