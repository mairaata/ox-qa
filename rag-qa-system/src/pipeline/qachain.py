from langchain.chains import RetrievalQA
from langchain import  PromptTemplate

class QAChain:
    DEFAULT_SYSTEM_PROMPT = """
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    """.strip()

    def __init__(self,llm_pipeline, retriever):
      self.retriever = retriever
      self.llm_pipeline = llm_pipeline

      self.SYSTEM_PROMPT = "Use the following context to answer the question at the end. Answer with reasoning. if you need to include table in your answer properly format it. If you don't know the answer, just say that you don't know, don't try to make up an answer."
      self.PROMPT =   """
              instruction: Please act as a sophisticated investment analyst. Give your analysis in a factual, non-speculative way based only on the content provided. Avoid any hypothetical assumptions or hallucinations.
          In your response, explicitly highlight all mentions of performance metrics like returns, percentages, time periods, charts/graphs etc using bold formatting. Also highlight other key sections like risks, fees, disclosures.
          Support your conclusion with specific evidence from the relative frequency, positioning and emphasis of performance information in the factsheet where necessary.
          {context}

          Question: {question}
          """
      self.prompt_Template = self.create_prompt_template()

      self.chain =  RetrievalQA.from_chain_type(
              llm=self.llm_pipeline.llm,
              chain_type="stuff",
              retriever=self.retriever.get_vectordb().as_retriever(search_kwargs={"k": 2}),
              return_source_documents=True,
              chain_type_kwargs={"prompt": self.prompt_Template},
          )


    def query(self, question):
        return self.chain(question)


    def generate_prompt(self,prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        return f"""
    [INST] <>
    {system_prompt}
    <>

    {prompt} [/INST]
    """.strip()


    def create_prompt_template(self):
      template = self.generate_prompt(self.PROMPT,system_prompt=self.SYSTEM_PROMPT)
      prompt = PromptTemplate(template=template, input_variables=["context", "question"])
      return prompt


