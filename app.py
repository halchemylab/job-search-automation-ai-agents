import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
import tempfile
from resume_parser_agent_openai import answer_query
from job_search_script import JobSearcher
from company_research_script import CompanyResearcher
import asyncio

st.title("AI Job Search Assistant")

# --- 1. Resume Upload and Parsing ---
st.header("1. Upload Your Resume")
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        resume_path = tmp_file.name

    st.success(f"Resume '{uploaded_file.name}' uploaded successfully.")

    # Use a button to trigger resume analysis
    if st.button("Analyze Resume"):
        with st.spinner("Analyzing resume..."):
            # Extract skills and experience from the resume
            skills_query = "Extract all technical skills, soft skills, and programming languages from the resume. List them as a comma-separated string."
            experience_query = "Summarize the candidate's work experience in 2-3 sentences."
            
            skills = answer_query(skills_query, resume_path)
            experience = answer_query(experience_query, resume_path)
            
            st.session_state.resume_data = {
                'skills': [skill.strip() for skill in skills.split(',')],
                'experience': [experience]
            }
            st.session_state.resume_analyzed = True

    if st.session_state.get('resume_analyzed'):
        st.subheader("Extracted Information:")
        st.write("**Skills:**", ", ".join(st.session_state.resume_data['skills']))
        st.write("**Experience Summary:**", st.session_state.resume_data['experience'][0])

# --- 2. Job Search Filters ---
st.header("2. Define Your Job Search")
if st.session_state.get('resume_analyzed'):
    location = st.text_input("Location", "Remote")
    job_types = st.multiselect("Job Type", ["full-time", "part-time", "contract", "internship"])
    work_style = st.selectbox("Work Style", ["Any", "On-site", "Remote", "Hybrid"])

    # --- 3. Search for Jobs ---
    if st.button("Search for Jobs"):
        st.header("3. Top Job Matches")
        with st.spinner("Searching for the latest job postings..."):
            filters = {
                "location": location,
                "job_types": job_types,
                "work_style": work_style
            }
            
            searcher = JobSearcher()
            jobs = searcher.search_jobs(st.session_state.resume_data, filters)
            st.session_state.jobs = jobs

# --- 4. Display Job Listings and Research ---
if st.session_state.get('jobs'):
    if not st.session_state.jobs:
        st.warning("No jobs found matching your criteria.")
    else:
        for i, job in enumerate(st.session_state.jobs):
            with st.expander(f"**{job.get('title', 'N/A')}** at **{job.get('company_name', 'N/A')}**"):
                st.write(f"**Location:** {job.get('location', 'N/A')}")
                st.write(f"**Salary:** {job.get('salary', 'Not specified')}")
                st.write(f"**Description:** {job.get('description', 'N/A')}")
                st.markdown(f"[View Job Posting]({job.get('url', '#')})", unsafe_allow_html=True)
                
                if st.button(f"Research {job.get('company_name', 'N/A')}", key=f"research_{i}"):
                    with st.spinner(f"Researching {job.get('company_name', 'N/A')}..."):
                        researcher = CompanyResearcher()
                        # Run the async function
                        company_info = asyncio.run(researcher.research_company(job.get('company_name', 'N/A')))
                        
                        st.subheader(f"Company Insights: {job.get('company_name', 'N/A')}")
                        st.write("**Company Overview:**", company_info.get('overview', 'Not available'))
                        st.write("**Recent Developments:**", company_info.get('recent_developments', 'Not available'))
                        st.write("**Culture and Benefits:**", company_info.get('culture_and_benefits', 'Not available'))

# --- Initial state setup ---
if 'resume_analyzed' not in st.session_state:
    st.session_state.resume_analyzed = False
if 'jobs' not in st.session_state:
    st.session_state.jobs = []
