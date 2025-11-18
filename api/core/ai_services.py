import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
load_dotenv(dotenv_path="scripts/.env") 

class AIService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set.")
        
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-pro-latest",  # Using your specified model
            google_api_key=self.api_key,
            temperature=0.2
        )

    def get_document_overview(self, document_text: str) -> str:
        """
        Generates a structured markdown analysis of the document.
        """
        
        # This is your new, better prompt from app.html
        prompt_template = """
        You are an expert AI paralegal named LawGeeks. Your specialty is analyzing legal documents and explaining them in simple, neutral terms for people who are not lawyers.
        Your analysis must be factual and objective. Do not provide legal advice, opinions, or any information not explicitly found in the document.

        Analyze the following document. Structure your response with the following headings EXACTLY as written, using "###" for each heading: ### Summary, ### Key Insights, ### Important Mentions, ### Vigilance Score (1-100) and Justification.
        For "Important Mentions" and "Key Insights", use bullet points starting with '*'.
        For "Important Mentions", extract and list all specific dates, deadlines, and financial amounts. If none are found, state "None found."
        For "Vigilance Score (1-100) and Justification", provide a numerical risk score from 1 (very low risk) to 100 (very high risk). After the score, provide a single sentence justifying your choice.

        Document:
        ---
        {document_text}
        """

        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        
        chain = prompt | self.llm | StrOutputParser()

        try:
            response = chain.invoke({"document_text": document_text})
            return response
        except Exception as e:
            print(f"Error in AI Service: {e}")
            return "### Error\n\nCould not generate analysis. The AI service failed."