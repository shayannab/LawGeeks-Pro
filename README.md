LawGeeks-Pro ⚖️

An intelligent paralegal assistant that simplifies and explains complex legal documents using Retrieval-Augmented Generation (RAG) and document analysis.

🚀 About The Project

Legal documents such as rental agreements, loan contracts, and terms of service—are often full of dense, technical language. This creates confusion and potential risks for individuals.

LawGeeks-Pro helps bridge this gap by offering a clear, private, and supportive environment to:

Generate simple, human-readable summaries.

Detect hidden risks, unfair clauses, and key financial terms.

Provide a Vigilance Score to rate potential risk (1–100).

Answer user questions about the document using a RAG system connected to a knowledge base of Indian law.

(Built for the “Generative AI for Demystifying Legal Documents” challenge.)

⚙️ Core Features

Multi-Format Upload: Analyze .pdf, .docx, and image (.png, .jpg) files, or paste text directly.

Automated Analysis: Generates a dashboard with summaries, insights, and extracted details (dates, amounts, clauses).

Vigilance Score: Dynamic meter to visualize document risk.

RAG-Based Q&A: Ask context-aware questions (e.g., “What if I miss a payment?”) and get answers grounded in your document and the legal knowledge base.

Professional Report: Download results as a clean PDF report.

Accessibility: Built-in text-to-speech and translation into multiple Indian languages.

🛠️ Tech Stack

Backend: FastAPI, Uvicorn

AI/LLM: Google Gemini (gemini-pro-latest)

RAG Framework: LangChain, ChromaDB, GoogleGenerativeAIEmbeddings

Frontend: HTML, TailwindCSS, Vanilla JavaScript

File Handling: pypdf, python-docx, tesseract

🏁 How to Run Locally
1. Prerequisites

Python 3.10 or higher

A Google Gemini API key

2. Setup

1. Clone the repository

git clone https://github.com/SIBAM890/LawGeeks-Pro.git
cd LawGeeks-Pro


2. Create and activate a virtual environment

# Windows
python -m venv .venv
.venv\Scripts\activate


3. Install dependencies

pip install -r requirements.txt


4. Add your environment variables

Go to the scripts/ folder and create a .env file:

GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"


5. Add your legal knowledge base

Place all relevant legal PDFs (e.g., Indian Contract Act, RERA guidelines) in the knowledge_base/ folder.

These files power the RAG system.

6. Build the vector database

cd scripts
python ingest.py


(Re-run if you add new documents.)

7. Start the application

cd ..
uvicorn api.index:app --reload

⚖️ Disclaimer

LawGeeks-Pro provides informational analysis only and should not be considered legal advice. Always consult a licensed legal professional for official guidance.


