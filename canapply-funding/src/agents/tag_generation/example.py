import asyncio

from src.config import settings  # noqa: F401 - loads .env
from src.agents.tag_generation.agent import TagGenerationAgent


content = {
    "full_name": "Ben Hamza",
    "department": "Computer Science",
    "institute": "Concordia University",
    "occupation": "Professor",
    "research_areas": ["Machine Learning", "Artificial Intelligence", "Computer Vision"],
    "area_of_expertise": ["Deep Learning", "Time Series Forecasting"],
    "credentials": "PhD in Computer Science",
    "url": "https://example.com/professor/benhamza"
}


async def main():
    generated_tags = await TagGenerationAgent().generate(content=content, regen=True)
    print("Generated Tags:\n", generated_tags)


if __name__ == "__main__":
    asyncio.run(main())
