import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(llm, portfolio, clean_text):
    # Page title and layout
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")

    # Header and description
    st.title("📧 Cold Mail Generator")

    # Add a reference to your GitHub portfolio
    st.markdown(
        """
        <h4 style="text-align: center;">👨‍💻 <a href="https://github.com/mirza7175/genai-cold-email-llm.git" target="_blank">Visit My GitHub Portfolio</a></h4>
        """,
        unsafe_allow_html=True,
    )

    # Project description
    st.markdown(
        """
        **Welcome to the Cold Mail Generator!**  
        This tool helps service companies streamline outreach by generating personalized cold emails for job opportunities.  

        ### **About This Project**
        - **Built With**: Groq, LangChain, Streamlit  
        - **Core Functionality**:  
            1. Extracts job descriptions from a provided URL (e.g., a company's careers page).  
            2. Generates tailored cold emails with links to relevant portfolio projects stored in a vector database.  

        ### **How It Works**:
        1. **Input a Job Posting URL**: Provide a valid URL from a company's careers page.  
        2. **Extract Job Descriptions**: The tool analyzes the content and identifies job details.  
        3. **Generate Cold Emails**: It creates professional, job-specific emails linking relevant portfolio projects.  

        _**Note:** Ensure the URL points to a page with job descriptions (e.g., a careers page or job listing)._  
        """
    )

    # Input field for the URL
    url_input = st.text_input(
        "Enter a Job Posting URL:",
        placeholder="e.g., https://example.com/careers/software-engineer"
    )

    # Submit button
    submit_button = st.button("Generate Cold Emails")

    # On button click
    if submit_button:
        if not url_input:
            st.warning("⚠️ Please enter a URL before submitting!")
        else:
            try:
                loader = WebBaseLoader([url_input])
                data = clean_text(loader.load().pop().page_content)
                portfolio.load_portfolio()
                jobs = llm.extract_jobs(data)

                if jobs:
                    st.success(f"✅ Found {len(jobs)} job(s). Generating emails...")
                    for job in jobs:
                        skills = job.get('skills', [])
                        links = portfolio.query_links(skills)
                        email = llm.write_mail(job, links)
                        st.code(email, language='markdown')
                else:
                    st.warning("⚠️ No job descriptions were found at the provided URL.")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    create_streamlit_app(chain, portfolio, clean_text)
