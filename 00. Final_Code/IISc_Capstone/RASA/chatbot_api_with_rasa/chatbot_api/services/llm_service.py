from langchain_community.llms import Ollama

llm = Ollama(model="mistral")

def query_llm(prompt: str ) -> str:
    response = llm.invoke(
                    prompt,
                    options={"num_predict": 300}
                )
    return response