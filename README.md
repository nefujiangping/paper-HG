# paper-HG
Codes for Graduation Project in NEFU -- Headline Generation of Paper Abstract. The core model is modified from abisee's [pointer-generator](https://github.com/abisee/pointer-generator).
The codes are divided into 4 parts in 4 folders, respectively.
- data: This part includes data(for train,val,test) and vocab(vocab, vocab_pos, vocab_ner).However, it only shows toy data.
- data-preprocess: Codes for preprocessing data.
- pointer-generator: Seq-to-Seq model for headline generation, modified from abisee's pointer-generator. Please go [here](https://github.com/abisee/pointer-generator) to see more details.
- post-analysis: Some scripts to analysis the decoded headline.
