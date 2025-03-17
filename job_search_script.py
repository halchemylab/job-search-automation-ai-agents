import os
from datetime import datetime, timedelta
from typing import List, Dict
import openai
from dotenv import load_dotenv

class JobSearcher:
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def search_jobs(self, resume_data: Dict, filters: Dict) -> List[Dict]:
        """
        Search for jobs based on resume data and user filters.
        """
        query = self._construct_search_query(resume_data, filters)
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",  # Primary model
                messages=[
                    {"role": "system", "content": "You are a job search assistant. Search for relevant jobs based on the candidate's profile and return exactly 3 most relevant positions from the last 24 hours."},
                    {"role": "user", "content": query}
                ],
                temperature=0.7
            )
            
            jobs = self._parse_job_listings(response.choices[0].message.content)
            return self._score_and_rank_jobs(jobs, resume_data)
        except Exception as e:
            print(f"Error in job search: {str(e)}")
            if "api_key" in str(e).lower():
                print("OpenAI API key error. Please check your .env file for a valid OPENAI_API_KEY.")
            return []

    def _construct_search_query(self, resume_data: Dict, filters: Dict) -> str:
        """Construct a search query based on resume data and filters"""
        skills = ', '.join(resume_data.get('skills', [])[:5])  # Top 5 skills
        experience = resume_data.get('experience', [])[0] if resume_data.get('experience') else ""
        
        query = f"""
        Find jobs matching this candidate profile:
        Skills: {skills}
        Recent Experience: {experience}
        Location: {filters.get('location', 'Any')}
        Remote: {'Yes' if filters.get('remote') else 'No preference'}
        Minimum Salary: ${filters.get('min_salary', 0):,}
        
        Please format each job as:
        Title: [job title]
        Company: [company name]
        Location: [location]
        Salary: [salary range if available]
        Description: [brief job description]
        URL: [job posting URL]
        """
        return query

    def _parse_job_listings(self, response_text: str) -> List[Dict]:
        """Parse the OpenAI response into structured job listings"""
        jobs = []
        current_job = {}
        
        # Split into individual job listings (splitting by numbered items)
        listings = response_text.split('\n\n')
        
        for listing in listings:
            if not listing.strip() or 'Title:' not in listing:
                continue
                
            current_job = {}
            lines = listing.split('\n')
            
            for line in lines:
                line = line.strip().replace('**', '')  # Remove markdown
                if ':' in line:
                    key, value = [part.strip() for part in line.split(':', 1)]
                    key = key.lower()
                    
                    # Handle URL markdown format
                    if key == 'url':
                        if '[' in value and '](' in value and ')' in value:
                            value = value[value.index('(')+1:value.index(')')]
                    
                    current_job[key] = value

            if current_job:
                jobs.append(current_job)
        
        return jobs

    def _score_and_rank_jobs(self, jobs: List[Dict], resume_data: Dict) -> List[Dict]:
        """Score and rank jobs based on match with resume"""
        for job in jobs:
            score = 0
            job_desc = job.get('description', '').lower()
            # Increase score for each matching skill
            for skill in resume_data.get('skills', []):
                if skill.lower() in job_desc:
                    score += 1
            # Increase score for experience keywords
            for exp in resume_data.get('experience', []):
                if any(keyword in job_desc for keyword in exp.lower().split()):
                    score += 0.5
            job['match_score'] = score
            
        return sorted(jobs, key=lambda x: x.get('match_score', 0), reverse=True)

def get_resume_data_from_input() -> Dict:
    """Collect resume data from terminal input."""
    skills_input = input("Enter your skills (comma-separated): ").strip()
    skills = [skill.strip() for skill in skills_input.split(',') if skill.strip()]

    experience_input = input("Enter your most recent experience: ").strip()
    experience = [experience_input] if experience_input else []
    
    return {
        'skills': skills,
        'experience': experience
    }

def get_filters_from_input() -> Dict:
    """Collect job search filters from terminal input."""
    location = input("Enter preferred location (or leave blank for 'Any'): ").strip() or "Any"
    
    remote_input = input("Are you open to remote positions? (yes/no): ").strip().lower()
    remote = True if remote_input in ['yes', 'y'] else False
    
    min_salary_input = input("Enter minimum salary (numbers only, or leave blank for 0): ").strip()
    min_salary = int(min_salary_input) if min_salary_input.isdigit() else 0
    
    return {
        "location": location,
        "remote": remote,
        "min_salary": min_salary
    }

def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
        return

    print("\n--- Job Search Assistant ---\n")
    resume_data = get_resume_data_from_input()
    filters = get_filters_from_input()

    searcher = JobSearcher()

    print("\nSearching for jobs...\n")
    results = searcher.search_jobs(resume_data, filters)
    
    print("Search Results:")
    print("==============")
    if not results:
        print("No matching jobs found.")
        return
    
    for job in results:
        print(f"\n🎯 {job.get('title', 'N/A')}")
        print(f"🏢 {job.get('company', 'N/A')}")
        print(f"📍 {job.get('location', 'N/A')}")
        print(f"💰 {job.get('salary', 'Not specified')}")
        print(f"🔗 {job.get('url', 'No URL provided')}")
        print(f"\nDescription:\n{job.get('description', 'N/A')}\n")
        print(f"Match Score: {job.get('match_score', 'N/A')}")
        print("-" * 50)

if __name__ == "__main__":
    main()
