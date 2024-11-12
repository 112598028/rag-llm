from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface.llms import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from backend.database_handler import ChromaDBHandler


model_name = "MediaTek-Research/Breeze-7B-Instruct-v1_0"
template = """
請根據以下上下文回答問題。如果無法使用提供的信息回答問題，請回答"我不知道"。

Context: {retrieve_data}

Question: {question}

Answer:
"""

class Retriever:
    def __init__(self, vectorstore, k=3):
        self.vectorstore = vectorstore
        self.k = k
        self.vectorstore = ChromaDBHandler()

    def retrieve_db(self, query):

        chromadb = self.vectorstore.get_chroma_db()
        retriever = chromadb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k},
        )
        results = retriever.invoke(query)
        return results


class Generator:
    def __init__(self, model_name=model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2" # optional
        )

        self.generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            # max_length=1024,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            do_sample=True
        )
        self.llm = HuggingFacePipeline(pipeline=self.generation_pipeline)
        self.output_parser = StrOutputParser()
        

    def generate_response(self, question, retrieve_data):
        prompt_template = PromptTemplate(
            input_variables=["question", "retrieve_data"],
            template=template,
        )
        prompt_text = prompt_template.format(question=question, retrieve_data=retrieve_data)
        response = self.llm.invoke(prompt_text)

        return self.output_parser.parse(response)


class RAG:
    def __init__(
        self,
        vectorstore,
        k=3,
        model_name=model_name,
        # temperature=0.0,
    ):
        self.retriever = Retriever(vectorstore, k)
        self.generator = Generator(model_name)

    def get_response(self, query):
        retrieve_data = self.retriever.retrieve_db(query)
        response = self.generator.generate_response(query, retrieve_data)
        return response
