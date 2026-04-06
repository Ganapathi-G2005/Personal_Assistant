import os
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

class M(BaseModel):
    a: str

try:
    res = ChatOpenAI(model='gpt-4.1-nano').with_structured_output(M).invoke('test')
    print("Success:", res)
except Exception as e:
    print("Error:", type(e), e)
