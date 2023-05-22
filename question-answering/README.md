# qa gen

This project consists of code for generating question answer pairs from a given text document.

## How does it work

- First generate N unique questions using `langchain` and `openai` (via prompt-engineering, see `qa_gen.py`).
- Secondly, use NASA v6 model (fine-tuned?) to generate answers for each of the questions generated (see `notebooks/cmr-bulk-qa-gen.ipynb`).

## Usage code example

### Build Question generator

```python
from qa_gen import (
    LangchainSimpleQuestionGenerator,
    CachedLangchainSimpleQuestionGenerator
)

# question_generator = LangchainSimpleQuestionGenerator(ChatOpenAI(temperature=0.0), n_questions=10)

# question_generator = CachedLangchainSimpleQuestionGenerator.load("<path to pickled file>")

question_generator = CachedLangchainSimpleQuestionGenerator(
    ChatOpenAI(temperature=0.0),
    n_questions=10,
)

# Or override default prompt
question_generator = CachedLangchainSimpleQuestionGenerator(
    ChatOpenAI(temperature=0.0),
    n_questions=10,
    prompt=<PROMPT_TEXT_TO_GENERATE_QUESTION_LIST>
)

document = <DOC_TEXT>
questions = question_generator.generate_questions_from_text(document)
```

### Build Answer generator

```python
from qa_gen import CachedLangchainAnswerGenerator


# override default prompt to generate answers
answer_generator = CachedLangchainAnswerGenerator(
    ChatOpenAI(temperature=0.0),
    prompt=<PROMPT_TEXT>,
)

answer_generator = CachedLangchainAnswerGenerator.load(<PICKLE_FILE>)

# generate answers from text for list of questions
answer_generator.generate_answers_from_text(
    document,
    questions,
)
```

### Build QA generator

```python
from qa_gen import (
    TransformerBasedQuestionAnswerGenerator,
    LangChainBasedQuestionAnswerGenerator,
    LangChainBasedQuestionAnswerGeneratorSpanable,
    LangChainBasedQuestionAnswerGeneratorSpanableRecursiveMatch,
)

# transformer based
qa_generator = TransformerBasedQuestionAnswerGenerator(
    question_generator=question_generator,
    model=model,
    tokenizer=tokenizer
)

# langchain (chatgpt based)
qa_generator = LangChainBasedQuestionAnswerGenerator(
    question_generator,
    answer_generator
)

# langchain (chatgpt based) but custom span
qa_generator = LangChainBasedQuestionAnswerGeneratorSpanable(
    question_generator,
    answer_generator
)

# Has custom span matching algorithm
qa_generator = LangChainBasedQuestionAnswerGeneratorSpanableRecursiveMatch(
    question_generator,
    answer_generator
)

qa_data = qa_generator.generate_qas_from_text(document)
```


### Bulk generation for CMR

```python
from tqdm import tqdm


def cmr_document_iterator(path):
    with open(path) as f:
        for data in json.load(f):
            text = data.get("text", "").strip()
            if not text:
                continue
            yield text

def bulk_qa_generator_2(doc_iterator, qa_generator, n_docs=10, cutoff_threshold=0.1):
    qas = []
    counter = 0
    for document in tqdm(doc_iterator, total=n_docs):
        qas.extend(qa_generator.generate_qas_from_text(document))
        counter += 1
        if counter >= n_docs:
            break
    return qas

def convert_to_sq2(data):
    res = dict(version="v2.0", data=[])
    for context, vals in groupby(data, key=lambda x: x["context"]):
        idx = str(hash(context))
        tmpdata = dict(title=idx, paragraphs=[dict(qas=[], context=context)])
        for _qad in vals:
            tmpdata["paragraphs"][0]["qas"].append(
                dict(
                    is_impossible="false",
                    question=_qad["question"],
                    answers=[dict(text=_qad["answer"], answer_start=_qad["start"])],
                    id=uuid.uuid4().hex,
                )
            )
        res["data"].append(tmpdata)
    return res

def bulk_qa_generator(doc_iterator, question_generator, qa_generator, n_docs=10, cutoff_threshold=0.1):
    qas = []
    counter = 0
    for document in tqdm(doc_iterator, total=n_docs):
        questions = question_generator.generate_questions_from_text(document)
        qa_data = qa_generator.generate_questions_from_text(
            document,
            cutoff_threshold=cutoff_threshold
        )
        qas.extend(qa_data)
        counter += 1
        if counter > n_docs:
            break
    return convert_to_sq2(qas)

question_generator = CachedLangchainSimpleQuestionGenerator(ChatOpenAI(temperature=0.0), n_questions=10)

qa_data = bulk_qa_generator(
    cmr_iterator,
    question_generator,
    qa_generator,
    n_docs=10
)
```
