import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from utils.store import Store

load_dotenv()


class LLM:
    llm = None
    EXPRESSIVE_CHAIN = None
    
    @classmethod
    def _initialize(cls):
        if cls.llm is not None:
            return
            
        try:
            if os.getenv("AZURE_OPENAI_API_KEY"):
                cls.llm = AzureChatOpenAI(
                    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                )
            else:
                cls.llm = ChatOpenAI(
                    model="gpt-4o",
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                )
            
            cls.EXPRESSIVE_PROMPT = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are meant to convert text from English to ASL Gloss grammar. Do not change meaning or move periods. I will send you a phrase, please rephrase it "
                        "it to follow ASL grammar order: object, then subject, then verb. Remove words like IS and ARE that are not present in ASL. "
                        "Replace I with ME. Do not add classifiers. Everything should be English text. Please output nothing but the rephrased phrase.",
                    ),
                    ("human", "{transcription}"),
                ]
            )
            cls.EXPRESSIVE_CHAIN = cls.EXPRESSIVE_PROMPT | cls.llm
        except Exception as e:
            print(f"Warning: Could not initialize LLM: {e}")
            cls.llm = None

    def gloss(self, transcription):
        """Convert English to ASL Gloss. Falls back to simple word splitting if LLM fails."""
        LLM._initialize()
        
        # Handle both string and list input
        if isinstance(transcription, list):
            raw_transcription = " ".join(transcription)
        else:
            raw_transcription = transcription
            
        # Try using LLM for proper ASL Gloss conversion
        if LLM.EXPRESSIVE_CHAIN is not None:
            try:
                response = LLM.EXPRESSIVE_CHAIN.invoke(
                    {
                        "transcription": raw_transcription,
                    }
                )
                result = response.content.strip()
                print(f"ASL Gloss: {result}")
                return result.split()
            except Exception as e:
                print(f"LLM error, falling back to simple split: {e}")
        
        # Fallback: simple word splitting without ASL grammar conversion
        print(f"Using fallback: {raw_transcription}")
        return raw_transcription.upper().split()
