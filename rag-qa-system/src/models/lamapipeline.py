from auto_gptq import AutoGPTQForCausalLM
from langchain import LLMChain, PromptTemplate, HuggingFacePipeline
from transformers import AutoTokenizer, TextStreamer, pipeline
from langchain.chains import RetrievalQA


from pipeline.rag import Reader

class LlamaPipeline:

    def __init__(self):
        self.model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
        self.model_basename = "model"
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.streamer =  TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.text_pipeline = self.create_text_pipeline()
        self.llm = self.create_llm_pipeline()

    def load_model(self):

      model = AutoGPTQForCausalLM.from_quantized(self.model_name_or_path, revision="gptq-4bit-128g-actorder_True",model_basename=self.model_basename,use_safetensors=True,trust_remote_code=True,inject_fused_attention=False,device=DEVICE,quantize_config=None,)
      return model

    def load_tokenizer(self):
       return  AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def create_text_pipeline(self):
      text_pipeline = pipeline(
        "text-generation",
        model=self.model,
        tokenizer=self.tokenizer,
        max_new_tokens=1024,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        streamer=self.streamer,
        )
      return text_pipeline

    def create_llm_pipeline(self):
      llm = HuggingFacePipeline(pipeline=self.text_pipeline, model_kwargs={"temperature": 0})
      return llm



class LlamaReader(Reader):
    def __init__(self,retriever):
       self.retriever=retriever
       self.llm_pipeline = LlamaPipeline()
       self.qa_chain = QAChain(llm_pipeline=self.llm_pipeline, retriever=self.retriever)

    def read(self, query):
       #return self.qa_chain.query(question=query)
        return self.qa_chain.query(query)





