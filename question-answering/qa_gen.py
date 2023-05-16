#!/usr/bin/env python3

from __future__ import annotations

import pickle
import re
import tempfile
import uuid

from itertools import groupby
from pathlib import Path
from pprint import pprint
from typing import List, Tuple, Optional, Union, Type
from abc import ABC, abstractmethod

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

    @property
    def __classname__(self) -> str:
        return self.__class__.__name__

    def save(self, path: Optional[str] = None):
        path = (
            tempfile.NamedTemporaryFile(prefix=self.__classname__, dir="tmp/").name
            if not path
            else path
        )
        logger.info(f"Saving instance of {self.__classname__} to {path}")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


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


class CachedLangchainAnswerGenerator(LangchainSimpleQuestionGenerator):
    """
    gpt-3.5-based question generator
    """

    _PROMPT_SYSTEM_QUESTION_SINGLE = "You are an expert user answering questions. You will be passed a page extracted from a documentation and a question. Generate a concise and informative answer to the question based *solely* on the given text. Keep the answer short and concise. Respond 'impossible_question' if not sure about the answer."
    # _PROMPT_SYSTEM_QUESTION_MULTI = "You are an expert user answering questions. You will be passed a text document and a list of questions related to the given document. Generate a numbered list of answers to each question based *solely* on the given text document. Each answers should be short and concise. Respond 'impossible_question' if not sure about the answer."
    _PROMPT_SYSTEM_QUESTION_MULTI = "You are an expert user answering questions. You will be passed a text document and a list of questions related to the given document. Generate a numbered list of answers to each question based *solely* on the given text document. Answer in an unbiased, comprehensive, and scholarly tone. Each answer should be very short, concise and informative. Respond 'impossible_question' if not sure about the answer. Responsd 'impossible_question'if the text context provides insufficient information."  # At the end of each answer, append a score on how confident you are on answering the question correctly in the given text context."

    def __init__(self, model: Optional = None) -> None:
        self.model = model
        self.cache = {}

    def generate_answers_from_text(self, text: str, questions: List[str]) -> List[str]:
        """
        This takes a list of questions and directly generates a list of answers
        in single API call.
        """
        text = text.strip()

        # convert a list of string to  single string in numbered bullets.
        question = "\n".join(
            map(lambda x: f"{x[0]}. {x[1]}", enumerate(questions, start=1))
        ).strip()

        messages = self._create_answer_extraction_conversation_messages(
            text=text,
            question=question,
            prompt=self._PROMPT_SYSTEM_QUESTION_MULTI,
        )

        # hash for tuple of (context, question) pair
        hsh = str(hash((text, question)))

        # if empty cache, run openai model
        answers = self.cache.get(hsh, [])
        answers = (
            self.bullets_to_list(self._run_model(messages)) if not answers else answers
        )

        self.cache[hsh] = answers

        return answers

    def generate_answer_from_text(self, text: str, question: str) -> str:
        """
        Generate an answer for a given text and question.
        """
        text = text.strip()
        messages = self._create_answer_extraction_conversation_messages(
            text=text,
            question=question,
            prompt=self._PROMPT_SYSTEM_QUESTION_SINGLE,
        )

        # hash for tuple of (context, question) pair
        hsh = str(hash((text, question)))

        # if empty cache, run openai model
        answers = self.cache.get(hsh, [])
        answers = self._run_model(messages) if not answers else answers

        self.cache[hsh] = answers

        return answers

    @staticmethod
    def _create_answer_extraction_conversation_messages(
        text: str,
        question: str,
        prompt: str,
    ) -> Tuple[SystemMessage, HumanMessage]:
        prompt = prompt.strip()
        question = question.strip()
        context_message = SystemMessage(content=prompt)

        # Create a human message containing the input text
        input_text_message = HumanMessage(content=text)

        input_question_message = HumanMessage(content=question)

        # Return the tuple of messages to be used in the extraction conversation
        return (context_message, input_text_message, input_question_message)


class QuestionAnswerGenerator(ABC):
    @abstractmethod
    def generate_qas_from_text(self, text: str, **kwargs) -> List[dict]:
        raise NotImplementedError()


class LangChainBasedQuestionAnswerGenerator(QuestionAnswerGenerator):
    def __init__(self, question_generator, answer_generator):
        self.question_generator = question_generator
        self.answer_generator = answer_generator

    def generate_qas_from_text(self, text: str) -> List[dict]:
        questions = self.question_generator.generate_questions_from_text(text)
        answers = self.answer_generator.generate_answers_from_text(text, questions)
        return list(
            map(
                lambda x: dict(
                    context=text,
                    start=-1,
                    end=-1,
                    score=None,
                    question=x[0],
                    answer=x[1],
                ),
                zip(questions, answers),
            )
        )


class TransformerBasedQuestionAnswerGenerator(QuestionAnswerGenerator):
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

    def generate_qas_from_text(
        self,
        text: str,
        cutoff_threshold: float = 0.4,
        remove_empty_answers: bool = True,
    ) -> List[dict]:
        questions = self.question_generator.generate_questions_from_text(text)

        data = self.__reformat_question_context(text, questions)

        return self.__infer_answers_from_questions(
            data,
            cutoff_threshold=cutoff_threshold,
            remove_empty_answers=remove_empty_answers,
        )

    @staticmethod
    def _convert_to_sq2(data):
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

    def __infer_answers_from_questions(
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
    def __reformat_question_context(context: str, questions: List[str]) -> List[dict]:
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
