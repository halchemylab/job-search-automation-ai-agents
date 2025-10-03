import asyncio
import aiohttp
from typing import Dict
import os
from dotenv import load_dotenv

class CompanyResearcher:
    def __init__(self):
        load_dotenv()
        self.rapidapi_key = os.getenv("RAPIDAPI_KEY")
        self.api_host = "real-time-glassdoor-data.p.rapidapi.com"

    async def research_company(self, company_name: str) -> Dict:
        url = f"https://{self.api_host}/company/search"
        headers = {
            "X-RapidAPI-Key": self.rapidapi_key,
            "X-RapidAPI-Host": self.api_host
        }
        params = {"query": company_name}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    # Assuming the first result is the most relevant
                    if data and data.get('data') and len(data['data']) > 0:
                        company_id = data['data'][0]['id']
                        return await self._get_company_details(session, company_id)
                    else:
                        return {"error": "Company not found"}
        except Exception as e:
            print(f"Error researching company {company_name}: {str(e)}")
            return {"error": "Unable to gather company information"}

    async def _get_company_details(self, session: aiohttp.ClientSession, company_id: str) -> Dict:
        url = f"https://{self.api_host}/company/reviews"
        headers = {
            "X-RapidAPI-Key": self.rapidapi_key,
            "X-RapidAPI-Host": self.api_host
        }
        params = {"id": company_id}

        try:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                if data and data.get('data'):
                    return self._normalize_company_data(data['data'])
                else:
                    return {"error": "No details found for company"}
        except Exception as e:
            print(f"Error getting company details: {str(e)}")
            return {"error": "Unable to fetch company details"}

    def _normalize_company_data(self, company_data: Dict) -> Dict:
        # For simplicity, we are just returning the raw data from the API
        # In a real application, you would want to extract and format the most important information
        return company_data

async def main():
    researcher = CompanyResearcher()
    try:
        company_name = input("Enter the company name to research: ")
        print(f"\nResearching {company_name}...\n")
        result = await researcher.research_company(company_name)
        
        print("Research Results:")
        print("=================")
        # The result is a dictionary containing the raw data from the API
        # You can process and display it as needed
        import json
        print(json.dumps(result, indent=2))

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        # Get only the tasks that aren't the current task
        current_task = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks() 
                if task is not current_task and not task.done()]
        
        if tasks:
            # Cancel pending tasks
            for task in tasks:
                task.cancel()
            # Wait for them to complete
            await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
