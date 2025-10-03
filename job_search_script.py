import os
from typing import List, Dict
import requests
from dotenv import load_dotenv

class JobSearcher:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("JSEARCH_API_KEY")
        self.api_url = "https://jsearch.p.rapidapi.com/search"

    def search_jobs(self, resume_data: Dict, filters: Dict) -> List[Dict]:
        """
        Search for jobs using the Jsearch API.
        """
        skills = ' '.join(resume_data.get('skills', [])[:5]) # Use top 5 skills for query
        query = f"{skills} in {filters.get('location', 'USA')}"
        print(f"Searching for: {query}") # for debugging

        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }
        
        params = {
            "query": query,
            "page": "1",
            "num_pages": "1",
            "country": "us",
            "language": "en",
            "date_posted": filters.get("date_posted", "all"),
            "employment_types": ",".join(filters.get("job_types", [])),
            "work_from_home": str(filters.get("work_style") == "Remote").lower(),
            "job_requirements": "no_experience",
            "radius": "1",
            "exclude_job_publishers": "BeeBe,Dice",
            "fields": "employer_name,job_publisher,job_title,job_country"
        }

        try:
            response = requests.get(self.api_url, headers=headers, params=params)
            response.raise_for_status()
            
            jobs = response.json().get('data', [])
            
            # The API returns a rich set of data, here we normalize it
            normalized_jobs = [self._normalize_job_data(job) for job in jobs]

            ranked_jobs = self._score_and_rank_jobs(normalized_jobs, resume_data)
            return ranked_jobs[:3] # Return top 3 jobs
            
        except requests.exceptions.RequestException as e:
            print(f"Error in job search: {str(e)}")
            return []

    def _normalize_job_data(self, job: Dict) -> Dict:
        """
        Normalize the job data from Jsearch API to a consistent format.
        """
        return {
            'title': job.get('job_title'),
            'company_name': job.get('employer_name'),
            'location': job.get('job_city', 'N/A') + ", " + job.get('job_state', 'N/A'),
            'description': job.get('job_description'),
            'url': job.get('job_apply_link'),
            'salary': f"{job.get('job_min_salary')} - {job.get('job_max_salary')}" if job.get('job_min_salary') else "Not specified"
        }

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

def main():
    # This is a sample execution for testing purposes.
    # In the actual application, resume_data and filters will be passed from the Streamlit app.
    resume_data = {
        'skills': ['Python', 'Data Analysis', 'Machine Learning'],
        'experience': ['2 years of experience in data science']
    }
    filters = {
        "location": "New York, NY",
        "job_types": ["full-time"],
        "work_style": "Any",
        "date_posted": "today"
    }

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
