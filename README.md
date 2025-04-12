# LLM Tutor Improvment

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

Conversation History: [(Tutor Q text, Student A text), (Tutor text, Student text), ...]
LLM Tutor Response

Mistake Identification (y1), (evaluated by accuracy and Macro F1)
Mistake Location (y2), (evaluated by accuracy and Macro F1)
Mistake Guidance (y3), (evaluated by accuracy and Macro F1)
Actionability (y4), (evaluated by accuracy and Macro F1)
LLM Tutor Name (y5), (evaluated by accuracy)

# TODO edit data-parser.py to chosen problem (pick one of the y variables to solve for)
