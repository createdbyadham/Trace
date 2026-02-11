import os
# CHANGE 1: Import AsyncOpenAI instead of OpenAI
from openai import AsyncOpenAI 
from dotenv import load_dotenv
from rag_service import RAGService
from memory_service import memory

load_dotenv()

class ReceiptAssistant:
    def __init__(self, rag_service: RAGService):
        # Initialize RAG service
        self.rag_service = rag_service
        
        # CHANGE 2: Use AsyncOpenAI client
        self.client = AsyncOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        
    async def ask_stream(self, query):
        
        # Get relevant context from RAG service
        context = await self.rag_service.get_relevant_context(query)

        # Get chat history from memory
        chat_history = memory.get_chat_history()
        
        system_prompt = f"""<system_prompt>
    <identity>
        <name>Trace</name>
        <creator>Adham Ehab</creator>
    </identity>
    <role>receipt_analysis_assistant</role>
    
    <style>
        <tone>Concise</tone>
        <tone>Direct</tone>
        <tone>No filler words</tone>
        <format>Bullet points or short sentences</format>
    </style>

    <instructions>
        <instruction>Answer the question immediately with data. Do not use opening phrases like "The discrepancy is due to..."</instruction>
        <instruction>Do NOT explain general concepts (e.g., do not explain what sales tax is or how retail works).</instruction>
        <instruction>For math questions, show the calculation only: "Item ($X) + Tax ($Y) = Total ($Z)".</instruction>
        <instruction>Keep responses under 50 words unless asked for a long list.</instruction>
        <instruction>You are a strict data analyst. Do not hallucinate items. Only use provided receipt_data.</instruction>
    </instructions>

    <context>
        <receipt_data>{context}</receipt_data>
    </context>
    
    <conversation_history>
        <chat>{chat_history if chat_history else 'No previous conversation'}</chat>
    </conversation_history>
</system_prompt>"""

        # Add user message to memory
        memory.add_user_message(query)

        # CHANGE 3: Await the creation of the stream
        stream = await self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            model="phi3.5",
            temperature=0.1,
            top_p=1.0,
            stream=True, # This stays True
            extra_body={
                "options": {
                    "num_ctx": 8192,
                    "num_gpu": 99,
                    "num_thread": 6
                },
                "keep_alive": -1
            }
        )

        full_response = ""
        
        # CHANGE 4: Use 'async for' to iterate over the stream
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                yield token

        # Save to memory
        memory.add_ai_message(full_response)