import asyncio
import aiohttp
import openai
from typing import Dict, Optional
import os
from dotenv import load_dotenv

class CompanyResearcher:
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if os.name == 'nt':  # Windows specific event loop policy
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def research_company(self, company_name: str) -> Dict:
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                tasks = [
                    self._get_company_overview(session, company_name),
                    self._get_recent_news(session, company_name),
                    self._get_employee_reviews(session, company_name)
                ]
                results = await asyncio.gather(*tasks)
            company_info = self._analyze_company_data(*results)
            return company_info

        except Exception as e:
            print(f"Error researching company {company_name}: {str(e)}")
            return {
                "company_name": company_name,
                "error": "Unable to gather company information",
                "status": "failed"
            }

    async def _get_company_overview(self, session: aiohttp.ClientSession, company_name: str) -> Dict:
        try:
            response = await openai.AsyncOpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a company research assistant. Provide key information about the company."},
                    {"role": "user", "content": f"Provide a brief overview of {company_name}, including industry, size, and key business areas. Summarize into 1-3 sentences. Don't output sources"}
                ]
            )
            return {"overview": response.choices[0].message.content}
        except Exception as e:
            print(f"Error in company overview: {str(e)}")
            if "api_key" in str(e).lower():
                return {"overview": "API key error - please check your OpenAI API key configuration"}
            return {"overview": f"Company overview not available: {str(e)}"}

    async def _get_recent_news(self, session: aiohttp.ClientSession, company_name: str) -> Dict:
        try:
            response = await openai.AsyncOpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a news research assistant. Provide recent developments about the company."},
                    {"role": "user", "content": f"What are the most significant recent developments or news about {company_name} in the last 6 months? Summarize into 1-3 sentences. Don't output Sources."}
                ]
            )
            return {"recent_news": response.choices[0].message.content}
        except Exception as e:
            print(f"Error in recent news: {str(e)}")
            if "api_key" in str(e).lower():
                return {"recent_news": "API key error - please check your OpenAI API key configuration"}
            return {"recent_news": f"Recent news not available: {str(e)}"}

    async def _get_employee_reviews(self, session: aiohttp.ClientSession, company_name: str) -> Dict:
        try:
            response = await openai.AsyncOpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a workplace culture analyst. Provide insights about the company's work culture."},
                    {"role": "user", "content": f"What is known about {company_name}'s work culture, benefits, and employee satisfaction? Summarize into 1-3 sentences. Don't output sources."}
                ]
            )
            return {"culture": response.choices[0].message.content}
        except Exception as e:
            print(f"Error in employee reviews: {str(e)}")
            if "api_key" in str(e).lower():
                return {"culture": "API key error - please check your OpenAI API key configuration"}
            return {"culture": f"Culture information not available: {str(e)}"}

    def _analyze_company_data(self, overview: Dict, news: Dict, culture: Dict) -> Dict:
        return {
            "overview": overview.get("overview", "Not available"),
            "recent_developments": news.get("recent_news", "Not available"),
            "culture_and_benefits": culture.get("culture", "Not available"),
            "analysis_timestamp": asyncio.get_event_loop().time(),
            "data_freshness": "24h"
        }

async def main():
    researcher = CompanyResearcher()
    try:
        company_name = input("Enter the company name to research: ")
        print(f"\nResearching {company_name}...\n")
        result = await researcher.research_company(company_name)
        
        print("Research Results:")
        print("=================")
        print("\n🏢Company Overview:")
        print(result.get('overview', 'Not available'))
        print("\n📍Recent Developments:")
        print(result.get('recent_developments', 'Not available'))
        print("\n💰Culture and Benefits:")
        print(result.get('culture_and_benefits', 'Not available'))
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
