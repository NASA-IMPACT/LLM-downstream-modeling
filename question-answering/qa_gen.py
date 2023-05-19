#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import pickle
import re
import string
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from itertools import groupby
from pathlib import Path
from pprint import pprint
from typing import List, Optional, Tuple, Type, Union

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.schema import HumanMessage, SystemMessage
from loguru import logger
from transformers import pipeline


def hash_string(string: str) -> str:
    # Create a hash object using the SHA-256 algorithm
    hash_object = hashlib.sha256()

    # Convert the string to bytes and update the hash object
    hash_object.update(string.encode("utf-8"))

    # Get the hexadecimal representation of the hash
    return hash_object.hexdigest()


class LangchainSimpleQuestionGenerator:
    """
    gpt-3.5-based question generator
    """

    # _PROMPT_SYSTEM_QUESTION = "You are an expert on extracting information from science documents to quiz people on a specific document. You will be passed a page extracted from the document. Write a numbered list of {} questions that can be answered based *solely* on the given text. Each question should be concise and should not consist of multiple parts. You can skip question if it has multiple parts. You can skip compound questions that have multiple questions in them. Only provide the questions that can be answered by a subject matter expert, who may not have access to the document."

    _PROMPT_SYSTEM_QUESTION = "I want you to act as a 100% accurate question-answering system that extracts information from scientific documents. You will be given a text document. Your tasks is to provide a detailed numbered list of {} relevant questions that can be answered *solely* on the given text document. Each questions should be concise, informative and complex. Refrain from combining multiple questions into single. Provide only questions that have scientific utility. Refrain from generic and factoid questions. Only provide the questions that can be answered by a subject matter expert, who may not have access to the document."
    _PROMPT_FOLLOWUP = "Refrain from combining multiple questions into a single one.  Provide only questions that have scientific utility.  Refrain from generic and factoid questions. Also don't generate questions that have long answers. Only generate questions that  have short answers that are substring in the text and can be answerable."

    _PATTERN_BULLET_TEXT = re.compile(r"^\s*\d+\.\s*(.+)$", re.MULTILINE)

    def __init__(
        self,
        model: Optional = None,
        n_questions: int = 10,
        prompt: str | None = None,
    ) -> None:
        self.model = model
        self.n_questions = n_questions
        self.prompt = prompt or LangchainSimpleQuestionGenerator._PROMPT_SYSTEM_QUESTION

    def generate_questions_from_text(self, text: str) -> list[str]:
        text = text.strip()
        messages = self._create_question_extraction_conversation_messages(
            text=text,
            question_prompt=self.prompt.format(self.n_questions),
        )
        output = self._run_model(messages)
        questions = self.bullets_to_list(output)

        # check if last question is complete
        if (len(questions) > 0) and (self.is_incomplete_text(questions[-1])):
            questions.pop()

        return questions

    def generate_questions_from_file(self, filepath: str | Path) -> list[str]:
        loader = TextLoader("data/test.md")
        doc = loader.load()[0]
        return self.generate_questions_from_text(doc.page_content)

    def _run_model(self, messages: tuple[SystemMessage, HumanMessage]) -> str:
        output = self.model._generate(messages)

        # Extract and return the generated text from the model output
        return output.generations[0].text.strip()

    @staticmethod
    def bullets_to_list(text: str) -> list[str]:
        text = text.strip()
        return LangchainSimpleQuestionGenerator._PATTERN_BULLET_TEXT.findall(text)

    @staticmethod
    def is_incomplete_text(text: str) -> bool:
        return not re.search(r"[.!?)]$", text.strip())

    @staticmethod
    def _create_question_extraction_conversation_messages(
        text: str,
        question_prompt: str,
    ) -> tuple[SystemMessage, HumanMessage]:
        question_prompt = question_prompt.strip()
        context_message = SystemMessage(content=question_prompt)

        # Create a human message containing the input text
        input_text_message = HumanMessage(content=text)

        input_followup_message = HumanMessage(
            content=CachedLangchainAnswerGenerator._PROMPT_FOLLOWUP,
        )

        # Return the tuple of messages to be used in the extraction conversation
        return (context_message, input_text_message, input_followup_message)

    @property
    def __classname__(self) -> str:
        return self.__class__.__name__

    def save(self, path: str | None = None):
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

    def __init__(
        self,
        model: Optional = None,
        n_questions: int = 10,
        prompt: str | None = None,
    ) -> None:
        super().__init__(model=model, n_questions=n_questions, prompt=prompt)
        self.cache = {}

    def generate_questions_from_text(self, text: str) -> list[str]:
        hsh = hash_string(text)
        questions = self.cache.get(hsh, []) or super().generate_questions_from_text(
            text,
        )

        self.cache[hsh] = questions

        return questions


class AnswerType(Enum):
    IMPOSSIBLE = 0
    INCOMPLETE = 1
    NO_SPAN = 2
    SPAN = 3


@dataclass
class AnswerDTO:
    answer: str
    context: str
    question: str
    answer_type: AnswerType
    raw_answer: str = None

    def __post_init__(self):
        if self.raw_answer is None:
            self.raw_answer = self.answer


@dataclass
class SpanAnswerDTO(AnswerDTO):
    start: int = -1
    end: int = -1
    score: float | None = None
    answer_type: AnswerType = field(default_factory=lambda: AnswerType.SPAN)


class CachedLangchainAnswerGenerator(LangchainSimpleQuestionGenerator):
    """
    gpt-3.5-based question generator
    """

    # _PROMPT_SYSTEM_QUESTION_MULTI = "You are an expert user answering questions. You will be passed a text document and a list of questions related to the given document. Generate a numbered list of answers to each question based *solely* on the given text document. Each answers should be short and concise. Respond 'impossible_question' if not sure about the answer."
    # _PROMPT_SYSTEM_QUESTION_MULTI = "You are an expert user answering questions. You will be passed a text document and a list of questions related to the given document. Generate a numbered list of answers to each question based *solely* on the given text document. Answer in an unbiased, comprehensive, and scholarly tone. Each answer should be very short, concise and informative. Respond 'impossible_question' if not sure about the answer. Responsd 'impossible_question'if the text context provides insufficient information."  # At the end of each answer, append a score on how confident you are on answering the question correctly in the given text context."
    _PROMPT_SYSTEM_QUESTION_MULTI = "I want you to act as a 100% accurate question-answering system that extracts information from scientific documents. You will be given a text document and a list of questions related to the given document. Your task is to generate a numbered list of answers to each question solely on the given text document. Generate only the texts that lie in the text document, and should be substrings in the text. Don't generate new words or make up words and don't add any extra words in the answer, and don't make up words. If possible, only generate answers as phrases, not complete sentences. For example, if the question is 'What is the year?', answer should be phrase only like '2023' instead of 'The year is 2023.'. Answer in an unbiased, comprehensive, and scholarly tone. Each answer should be very short, concise and informative. Respond 'impossible_question' if not sure about the answer. Respond 'impossible_question' if the text context provides insufficient information."

    # _PROMPT_FOLLOWUP = "Please, fefrain from completing the sentence. Just give me answers at phrase level. These phrases should be substring from the document. Don't generate new words or make up words and don't add any extra words to the answer, and don't mak up words."
    _PROMPT_FOLLOWUP = "Strictly, refrain from completing the sentence. Just give me answers at the phrase level. These phrases should be substring from the document. Don't generate new words or make up words and don't add any extra words to the answer, and don't make up words. "

    def __init__(self, model: Optional = None, prompt: str | None = None) -> None:
        self.model = model
        self.cache = {}
        self.prompt = (
            prompt or CachedLangchainAnswerGenerator._PROMPT_SYSTEM_QUESTION_MULTI
        )

    def generate_answers_from_text(
        self,
        text: str,
        questions: list[str],
    ) -> list[AnswerDTO]:
        """
        This takes a list of questions and directly generates a list of answers
        in single API call.
        """
        text = text.strip()

        # convert a list of string to  single string in numbered bullets.
        # question = "\n".join(
        #     map(lambda x: f"{x[0]}. {x[1]}", enumerate(questions, start=1)),
        # ).strip()
        question = "\n".join(questions).strip()

        messages = self._create_answer_extraction_conversation_messages(
            text=text,
            question=question,
            prompt=self.prompt,
        )

        # hash for tuple of (context, question) pair
        hsh = hash_string(f"{text}|question")

        # if empty cache, run openai model
        answers = self.cache.get(hsh, [])
        answers = (
            self.bullets_to_list(self._run_model(messages)) if not answers else answers
        )

        # convert strings to answer dto
        res = list(
            map(
                lambda x: self._convert_to_dto(
                    context=text,
                    question=x[0],
                    answer=x[1],
                ),
                zip(questions, answers),
            ),
        )

        self.cache[hsh] = res

        return res

    def _convert_to_dto(self, context: str, question: str, answer: str) -> AnswerDTO:
        # default to NO_SPAN type
        answer_dto = (
            answer
            if isinstance(answer, AnswerDTO)
            else AnswerDTO(
                answer=answer,
                answer_type=AnswerType.NO_SPAN,
                context=context,
                question=question,
            )
        )
        if LangchainSimpleQuestionGenerator.is_incomplete_text(answer_dto.answer):
            answer_dto.answer_type = AnswerType.INCOMPLETE
        elif "impossible_question" in answer_dto.answer.lower():
            answer_dto.answer_type = AnswerType.IMPOSSIBLE
        return answer_dto

    @staticmethod
    def _create_answer_extraction_conversation_messages(
        text: str,
        question: str,
        prompt: str,
    ) -> tuple[SystemMessage, HumanMessage]:
        prompt = prompt.strip()
        context_message = SystemMessage(content=prompt)

        # input_question_message = HumanMessage(content=f"Questions: {question}")
        input_question_message = HumanMessage(content=question)

        # input_text_message = HumanMessage(content=f"Document: {text}")
        input_text_message = HumanMessage(content=text)

        input_followup_message = HumanMessage(
            content=CachedLangchainAnswerGenerator._PROMPT_FOLLOWUP,
        )

        # Return the tuple of messages to be used in the extraction conversation
        return (
            context_message,
            input_text_message,
            input_question_message,
            input_followup_message,
        )


class QuestionAnswerGenerator(ABC):
    @abstractmethod
    def generate_qas_from_text(self, text: str, **kwargs) -> list[type[AnswerDTO]]:
        raise NotImplementedError()


class LangChainBasedQuestionAnswerGenerator(QuestionAnswerGenerator):
    def __init__(self, question_generator, answer_generator):
        self.question_generator = question_generator
        self.answer_generator = answer_generator

    def generate_qas_from_text(self, text: str) -> list[type[AnswerDTO]]:
        questions = self.question_generator.generate_questions_from_text(text)
        answers = self.answer_generator.generate_answers_from_text(text, questions)
        return answers


class LangChainBasedQuestionAnswerGeneratorSpanable(QuestionAnswerGenerator):
    """
    This QA generator does simple substring matching to figure out start/end
    values for the answer.
    """

    def __init__(self, question_generator, answer_generator):
        self.question_generator = question_generator
        self.answer_generator = answer_generator

    def generate_qas_from_text(self, text: str) -> list[dict]:
        questions = self.question_generator.generate_questions_from_text(text)
        answers = self.answer_generator.generate_answers_from_text(text, questions)

        return list(
            map(self.compute_spans, answers),
        )

    def matcher(self, context: str, answer: str):
        answer = answer.strip().rstrip(string.punctuation)
        try:
            return re.search(rf"\b{answer}\b", context, flags=re.IGNORECASE)
        except:
            answer = re.escape(answer)
            return re.search(rf"\b{answer}\b", context, flags=re.IGNORECASE)

    def compute_spans(self, answer: AnswerDTO) -> type[AnswerDTO]:
        match_ = self.matcher(answer.context, answer.raw_answer)

        # polymorphism-ish
        answer = SpanAnswerDTO(**answer.__dict__)

        # if match, just update the fields
        if match_:
            start = match_.start()
            end = match_.end()
            text = match_.group()

            # remove punctuation from the right side and update the span
            text_s = text.rstrip(string.punctuation + " ").rstrip()
            end -= len(text) - len(text_s)

            answer.answer = text_s
            answer.answer_type = AnswerType.SPAN
            answer.start = start
            answer.end = end
            answer.score = 1.0
        else:
            answer.answer_type = AnswerType.NO_SPAN
        return answer


class LangChainBasedQuestionAnswerGeneratorSpanableRecursiveMatch(
    LangChainBasedQuestionAnswerGeneratorSpanable,
):
    """
    This QA generator does simple substring matching to figure out start/end
    values for the answer.

    In cases where the LLM has appeneded preposition like "to", etc.
        do some strategic search to get the spans.
    """

    def __init__(self, question_generator, answer_generator, min_tokens: int = 2):
        super().__init__(question_generator, answer_generator)
        self.min_tokens = min_tokens

    def matcher(self, context: str, answer: str):
        match_ = super().matcher(context, answer)
        if match_:
            return match_
        tokens = answer.split(" ")
        while not match_ and len(tokens) >= self.min_tokens + 1:
            tokens = tokens[1:]
            match_ = super().matcher(context, " ".join(tokens))
        return match_


class TransformerBasedQuestionAnswerGenerator(QuestionAnswerGenerator):
    def __init__(
        self,
        question_generator: type[LangchainSimpleQuestionGenerator],
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
    ) -> list[SpanAnswerDTO]:
        questions = self.question_generator.generate_questions_from_text(text)

        data = self.__reformat_question_context(text, questions)

        data = self.__infer_answers_from_questions(
            data,
            cutoff_threshold=cutoff_threshold,
            remove_empty_answers=remove_empty_answers,
        )
        return data

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
                    ),
                )
            res["data"].append(tmpdata)
        return res

    def __infer_answers_from_questions(
        self,
        data: list[dict],
        cutoff_threshold,
        remove_empty_answers: bool,
    ) -> list[SpanAnswerDTO]:
        answers = self.qa_pipe(data)
        answers = map(lambda x: {**x, **dict(answer_type=AnswerType.SPAN)}, answers)
        res = filter(lambda x: x[1]["score"] >= cutoff_threshold, zip(data, answers))
        res = (
            filter(lambda x: len(x[1]["answer"].strip()) > 0, res)
            if remove_empty_answers
            else res
        )
        data = map(lambda x: {**x[0], **x[1]}, res)
        data = map(lambda x: SpanAnswerDTO(**x, text=x["answer"]), data)
        return list(data)

    @staticmethod
    def __reformat_question_context(context: str, questions: list[str]) -> list[dict]:
        return list(map(lambda x: dict(context=context, question=x), questions))


def main():
    qa_gen = LangchainSimpleQuestionGenerator(
        ChatOpenAI(temperature=0.0),
        n_questions=3,
    )

    # questions = qa_gen.generate_questions_from_text(doc.page_content)
    questions = qa_gen.generate_questions_from_file("data/test.md")

    print(len(questions))
    pprint(questions)


if __name__ == "__main__":
    main()
