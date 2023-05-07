#!/usr/bin/env python3

from __future__ import annotations

import re
import uuid

from itertools import groupby
from pathlib import Path
from pprint import pprint
from typing import List, Tuple, Optional, Union, Type

from transformers import pipeline

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.schema import HumanMessage, SystemMessage

from loguru import logger


class LangchainSimpleQuestionGenerator:
    """
    gpt-3.5-based question generator
    """

    _PROMPT_SYSTEM_QUESTION = "You are an expert on extracting information from science documents to quiz people on a specific document. You will be passed a page extracted from the document. Write a numbered list of {} questions that can be answered based *solely* on the given text. Each question should be concise and should not consist of multiple parts. You can skip question if it has multiple parts. You can skip compound questions that have multiple questions in them. Only provide the questions that can be answered by a subject matter expert, who may not have access to the document."

    _PATTERN_BULLET_TEXT = re.compile(r"^\s*\d+\.\s*(.+)$", re.MULTILINE)

    def __init__(self, model: Optional = None, n_questions: int = 10) -> None:
        self.model = model
        self.n_questions = n_questions

    def generate_questions_from_text(self, text: str) -> List[str]:
        text = text.strip()
        messages = self._create_question_extraction_conversation_messages(
            text=text,
            question_prompt=self._PROMPT_SYSTEM_QUESTION.format(self.n_questions),
        )
        output = self._run_model(messages)
        return self.bullets_to_list(output)

    def generate_questions_from_file(self, filepath: Union[str, Path]) -> List[str]:
        loader = TextLoader("data/test.md")
        doc = loader.load()[0]
        return self.generate_questions_from_text(doc.page_content)

    def _run_model(self, messages: Tuple[SystemMessage, HumanMessage]) -> str:
        output = self.model._generate(messages)

        # Extract and return the generated text from the model output
        return output.generations[0].text.strip()

    @staticmethod
    def bullets_to_list(text: str) -> List[str]:
        text = text.strip()
        questions = LangchainSimpleQuestionGenerator._PATTERN_BULLET_TEXT.findall(text)

        # Check if the last question is incomplete (does not end with punctuation or a parenthesis)
        if (len(questions) > 0) and (not re.search(r"[.!?)]$", questions[-1].strip())):
            questions.pop()
        return questions

    @staticmethod
    def _create_question_extraction_conversation_messages(
        text: str,
        question_prompt: str,
    ) -> Tuple[SystemMessage, HumanMessage]:
        question_prompt = question_prompt.strip()
        context_message = SystemMessage(content=question_prompt)

        # Create a human message containing the input text
        input_text_message = HumanMessage(content=text)

        # Return the tuple of messages to be used in the extraction conversation
        return (context_message, input_text_message)


class CachedLangchainSimpleQuestionGenerator(LangchainSimpleQuestionGenerator):
    """
    gpt-3.5-based question generator
    """

    def __init__(self, model: Optional = None, n_questions: int = 10) -> None:
        self.model = model
        self.n_questions = n_questions
        self.cache = {}

    def generate_questions_from_text(self, text: str) -> List[str]:
        text = text.strip()
        messages = self._create_question_extraction_conversation_messages(
            text=text,
            question_prompt=self._PROMPT_SYSTEM_QUESTION.format(self.n_questions),
        )
        hsh = str(hash(text))
        questions = self.cache.get(hsh, [])

        # if empty cache, run openai model
        questions = (
            self.bullets_to_list(self._run_model(messages))
            if not questions
            else questions
        )
        self.cache[hsh] = questions

        return questions


class QuestionAnswerGenerator:
    def __init__(
        self,
        question_generator: Type[LangchainSimpleQuestionGenerator],
        model,
        tokenizer,
        device: str = "cpu",
    ):
        self.question_generator = question_generator
        self.qa_pipe = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            handle_impossible_answer=True,
            device=device,
        )

    def generate_questions_from_text(
        self,
        text: str,
        cutoff_threshold: float = 0.4,
        remove_empty_answers: bool = True,
    ) -> List[dict]:
        questions = self.question_generator.generate_questions_from_text(text)
        data = self._reformat_question_context(text, questions)
        return self._infer_answers_from_questions(
            data,
            cutoff_threshold=cutoff_threshold,
            remove_empty_answers=remove_empty_answers,
        )

    @staticmethod
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

    def _infer_answers_from_questions(
        self,
        data: List[dict],
        cutoff_threshold,
        remove_empty_answers: bool,
    ):
        answers = self.qa_pipe(data)
        res = filter(lambda x: x[1]["score"] >= cutoff_threshold, zip(data, answers))
        res = (
            filter(lambda x: len(x[1]["answer"].strip()) > 0, res)
            if remove_empty_answers
            else res
        )
        return list(map(lambda x: {**x[0], **x[1]}, res))

    @staticmethod
    def _reformat_question_context(context: str, questions: List[str]) -> List[dict]:
        return list(map(lambda x: dict(context=context, question=x), questions))


def main():
    qa_gen = LangchainSimpleQuestionGenerator(
        ChatOpenAI(temperature=0.0), n_questions=3
    )

    # questions = qa_gen.generate_questions_from_text(doc.page_content)
    questions = qa_gen.generate_questions_from_file("data/test.md")

    print(len(questions))
    pprint(questions)


if __name__ == "__main__":
    main()
