import json
import traceback
from typing import List, Dict
from groq import Groq
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone


class RAGPipeline:
    def __init__(
        self,
        embedding_model_name="all-MiniLM-L6-v2",
        pinecone_api_key=None,
        pinecone_env=None,
        groq_api_key=None,
    ):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
        self.index = self.pc.Index("legal-llm")

        # Initialize Groq client
        self.client = Groq(api_key=groq_api_key)

    def query_pinecone(self, prompt: str, top_k: int = 10) -> List[Dict]:
        """Vector search with metadata filtering and scoring"""
        query_embedding = self.embedding_model.encode(prompt).tolist()

        query_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={
                # Optional: Add metadata filtering logic
                # "domain": "legal"
            },
        )

        results = [
            {
                "text": match.metadata["text"],
                "score": match.score,
                "metadata": match.metadata,
            }
            for match in query_results.matches
        ]

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def get_context(self, user_prompt: str, top_k: int = 3) -> str:
        """Retrieve and process top contextual results"""
        vector_results = self.query_pinecone(user_prompt, top_k)

        return "\n\n".join([result["text"] for result in vector_results])

    def generate_response(self, user_query: str, context: str) -> Dict:
        """Response generation with structured output"""
        system_prompt = f"""
        You are an expert legal analyst. Provide precise, evidence-based responses.

        Context: {context}
        
        Response Guidelines:
        - Analyze the query using ONLY the provided context
        - Structure response as JSON with:
          1. "answer": Comprehensive legal explanation
          2. "reasoning": Logical breakdown
          3. "confidence_score": 0-1 rating
          4. "key_sources": Relevant context snippets
        - Be concise but thorough
        - Explicitly state if context is insufficient
        """

        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                max_tokens=2048,
                temperature=0.3,
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            return {"error": str(e), "trace": traceback.format_exc()}
