# qa gen

This project consists of code for generating question answer pairs from a given text document.

## How does it work

- First generate N unique questions using `langchain` and `openai` (via prompt-engineering, see `qa_gen.py`).
- Secondly, use NASA v6 model (fine-tuned?) to generate answers for each of the questions generated (see `notebooks/cmr-bulk-qa-gen.ipynb`).

## Usage code example

### Simple generation

```python
from qa_gen import (
    LangchainSimpleQuestionGenerator,
    CachedLangchainSimpleQuestionGenerator,
    QuestionAnswerGenerator
)

# question-only generator

# question_generator = LangchainSimpleQuestionGenerator(ChatOpenAI(temperature=0.0), n_questions=10)
question_generator = CachedLangchainSimpleQuestionGenerator(ChatOpenAI(temperature=0.0), n_questions=10) # avoids repetitive openai call for same document

questions = question_generator.generate_questions_from_text(document)

qa_generator = QuestionAnswerGenerator(
    question_generator=question_generator,
    model=model,
    tokenizer=tokenizer
)

qa_data = qa_generator.generate_questions_from_text(document, cutoff_threshold=0.1)

```


### Bulk generation

```python
from qa_gen import (
    LangchainSimpleQuestionGenerator,
    CachedLangchainSimpleQuestionGenerator,
    QuestionAnswerGenerator
)
from tqdm import tqdm


def cmr_document_iterator(path):
    with open(path) as f:
        for data in json.load(f):
            text = data.get("text", "").strip()
            if not text:
                continue
            yield text

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

qa_generator = QuestionAnswerGenerator(
    question_generator=question_generator,
    model=model,
    tokenizer=tokenizer
)

cmr_qa_sq2 = bulk_qa_generator(
    cmr_iterator,
    question_generator,
    qa_generator,
    n_docs=10
)

with open("data/cmr_qa_sqv2.json", "w") as f:
    json.dump(cmr_qa_sq2, f)


```
