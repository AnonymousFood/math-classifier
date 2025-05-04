# LLM Tutor Improvement

### How is the data arranged

Conversations are split into pairs of two, (Tutor, Student) pairs.

The first entry is a special case of the initial question and answer pairs and are labeled accorrdingly
First Tutor could Be noted [TUTOR] [INITIAL QUESTION]
First Student could be noted [STUDENT] [INITIAL ANSWER]

The rest are labeled as (Tutor, Student) pairs with the follow up tag.
Second Tutor could be noted [TUTOR] [FOLLOW UP]
Second Student could be noted [STUDENT] [FOLLOW UP]

This is because these two parts of the conversation are different enough to explicity classify them.

### Data Structure Used:

Trained 3 transformer models (BERT, T5, GPT-2) to predict these 2 metrics:

Mistake Identification (y1), (evaluated by Weighted F1)
Actionability (y4), (evaluated by Weighted F1)