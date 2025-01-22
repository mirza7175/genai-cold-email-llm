import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
           
You are Mirza, a Business Development Executive at SmartPath AI.
SmartPath AI is an AI & Software Consulting company specializing in transforming business processes with cutting-edge AI-driven automation tools. With a proven track record of empowering enterprises, SmartPath AI delivers tailored solutions that enhance scalability, optimize processes, reduce costs, and improve overall efficiency.

Your task is to craft a cold email to a potential client based on the job details provided. Highlight how SmartPath AI’s expertise aligns with their specific requirements and describe the company’s capability to fulfill their needs. Include links to the most relevant projects or case studies from SmartPath AI's portfolio, sourced from the following list: {link_list}.

Write a professional, concise, and persuasive email.
Focus on showcasing SmartPath AI’s value proposition and its ability to deliver impactful results for the client.
Personalize the email based on the client’s job requirements. keep it concise.
Note: Do not include a preamble or introduce yourself outside the email body
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))