import asyncio

from src.config import settings  # noqa: F401 - loads .env
from src.agents.detailed_generation.agent import DetailedEmailGenerationAgent


tpl = """
Dear Professor {{ProfessorName}},

My name is Behnam Farsi, and I am reaching out because I believe I can meaningfully contribute to your research group. I have reviewed your recent work, and it aligns closely with my academic background and research interests.

I spent some time reading your research publications and came across your paper “{{PaperTitle}}”, published in {{Year}} in {{JournalName}}. {{RESEARCH_CONNECTION}}. I would be happy to share some of my insights on this work, which I believe could be a good starting point if I have the opportunity to join your team.

I completed my Masters at Concordia (CGPA: 3.8/4.0), with a thesis titled “Load Forecasting”. My studies are focused on Machine learning, AI, which strongly overlap with your work in {{ProfessorInterests}}. I am highly motivated to deepen my expertise by joining your lab as a PhD student.

I have attached my CV for your review. I would appreciate knowing if you are currently accepting new PhD students, and I would be happy to meet online to discuss how I could contribute to your team.

Thank you for your time and consideration. I look forward to hearing from you.

Best regards,
Behnam Farsi
"""

profiles = [
    {
        "prof": {
            "last_name": "Ben Hamza"
        },
        "template": {"template": tpl},
        "row": {
            "id": 12345,
            "funding_request_id": "12345",
            "research_interest": "Computer Vision; Artificial Intelligence",
            "paper_title": "A federated large language model for long-term time series forecasting",
            "research_connection": "Long-term time series forecasting in centralized environments poses unique challenges regarding data privacy, communication overhead, and scalability. To address these challenges, we propose FedTime, a federated large language model (LLM) tailored for long-range time series prediction. Specifically, we introduce a federated pre-trained LLM with fine-tuning and alignment strategies. Prior to the learning process, we employ K-means clustering to partition edge devices or clients into distinct clusters, thereby facilitating more focused model training. We also incorporate channel independence and patching to better preserve local semantic information, ensuring that important contextual details are retained while minimizing the risk of information loss. We demonstrate the effectiveness of our FedTime model through extensive experiments on various real-world forecasting benchmarks, showcasing substantial improvements over recent approaches. In addition, we demonstrate the efficiency of FedTime in streamlining resource usage, resulting in reduced communication overhead.",
            "year": 2022,
            "journal": "Journal of AI Research"
        },
    },
]

async def main():
    agent = DetailedEmailGenerationAgent()
    for profile in profiles:
        email = await agent.generate(
            row=profile["row"],
            prof=profile["prof"],
            template=profile["template"],
            regen=True
        )
        print(f"Generated email for funding_request_id={profile['row']['funding_request_id']}:\n")
        print(email)
        print("\n" + "= " *80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
