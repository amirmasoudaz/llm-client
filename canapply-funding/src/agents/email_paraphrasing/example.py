import asyncio

from src.config import settings  # noqa: F401 - loads .env
from src.agents.email_paraphrasing.agent import ParaphrasingAgent


content = """Dear Professor Ben Hamza,

My name is Behnam Farsi, and I am reaching out because I believe I can meaningfully contribute to your research group. I have reviewed your recent work, and it aligns closely with my academic background and research interests.

I spent some time reading your research publications and came across your paper “A federated large language model for long-term time series forecasting”, published in 2022 in Journal of AI Research. I am particularly interested in how FedTime leverages K-means client clustering and channel-independent patching to preserve local semantics while reducing communication, which seems promising for privacy-preserving forecasting in domains like energy and healthcare. I would be eager to explore extending these ideas to high-dimensional spatiotemporal data and vision-centric sensing scenarios as part of a PhD project. 

Regarding my background, I completed my Masters at Concordia (CGPA: 3.8/4.0), with a thesis titled “Load Forecasting”. My studies are focused on Machine learning, AI, which strongly overlap with your work in Computer Vision; Artificial Intelligence. I am highly motivated to deepen my expertise by joining your lab as a PhD student.

I’ve attached my CV for your review. I’d be happy to share my thoughts on this work, which I believe could be a good starting point if I have the chance to join your team. I would also appreciate knowing if you are currently accepting new PhD students, and I’d be glad to meet online to talk about how I could contribute.

Thank you for your time and consideration. I look forward to hearing from you.

Best regards,
Behnam Farsi"""


async def main():
    agent = ParaphrasingAgent()
    revised_email = await agent.paraphrase(content=content)
    print(f"Revised Email:\n{revised_email}")


if __name__ == "__main__":
    asyncio.run(main())
