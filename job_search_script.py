import os
from datetime import datetime, timedelta
from typing import List, Dict
import requests
from dotenv import load_dotenv

class JobSearcher:
    def __init__(self):
        load_dotenv()

    def search_jobs(self, resume_data: Dict, filters: Dict) -> List[Dict]:
        """
        Search for jobs using the Arbeitnow API.
        """
        skills = ', '.join(resume_data.get('skills', [])[:5])
        location = filters.get('location', 'Any')
        
        # The API doesn't have a specific location filter, so we include it in the search query
        search_query = f"{skills} {location}"

        try:
            response = requests.get(
                "https://www.arbeitnow.com/api/job-board-api",
                params={"search": search_query, "page": 1}
            )
            response.raise_for_status()  # Raise an exception for bad status codes
            
            jobs = response.json().get('data', [])
            
            # Filter jobs posted in the last 24 hours
            recent_jobs = self._filter_recent_jobs(jobs)
            
            return self._score_and_rank_jobs(recent_jobs, resume_data)
        except requests.exceptions.RequestException as e:
            print(f"Error in job search: {str(e)}")
            return []

    def _filter_recent_jobs(self, jobs: List[Dict]) -> List[Dict]:
        """Filters jobs to include only those posted in the last 24 hours."""
        recent_jobs = []
        twenty_four_hours_ago = datetime.utcnow() - timedelta(days=1)
        
        for job in jobs:
            created_at_str = job.get('created_at')
            if created_at_str:
                # The API returns a Unix timestamp in milliseconds
                created_at = datetime.utcfromtimestamp(int(created_at_str) / 1000)
                if created_at >= twenty_four_hours_ago:
                    recent_jobs.append(job)
        return recent_jobs

    def _score_and_rank_jobs(self, jobs: List[Dict], resume_data: Dict) -> List[Dict]:
        """Score and rank jobs based on match with resume"""
        for job in jobs:
            score = 0
            job_desc = job.get('description', '').lower()
            # Increase score for each matching skill
            for skill in resume_data.get('skills', []):
                if skill.lower() in job_desc:
                    score += 1
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
        print(f"🏢 {job.get('company_name', 'N/A')}")
        print(f"📍 {job.get('location', 'N/A')}")
        print(f"💰 {job.get('salary', 'Not specified')}")
        print(f"🔗 {job.get('url', 'No URL provided')}")
        print(f"\nDescription:\n{job.get('description', 'N/A')}\n")
        print(f"Match Score: {job.get('match_score', 'N/A')}")
        print("-" * 50)

if __name__ == "__main__":
    main()
