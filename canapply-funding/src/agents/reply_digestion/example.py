import asyncio
import json
from collections import Counter

import pandas as pd

from src.agents.reply_digestion.agent import ReplyDigestionAgent
from src.db.session import DB


async def agent_test():
    agent = ReplyDigestionAgent()

    main_email_body = """Dear Prof. Koo,  I hope this message finds you well. My name is Farnaz Masoumi, and I hold a Master's degree in Marketing Management from Central Tehran Azad University. I am writing to express my interest in pursuing a Master/PhD position under your supervision, focusing on consumer behaviour and customer experience.  I have been following your research with great interest, particularly your 2024 article titled "The Impact of Generative AI on Syllabus Design and Learning," published in the Journal of Marketing Education. Your paper's insights into students’ preference for AI-generated syllabi—especially its connection to perceptions of objectivity—offer an exciting perspective on how technology can influence educational experiences. With my background in consumer behavior and customer experience, I am eager to explore how similar mechanisms of perceived objectivity might affect consumer trust and engagement in AI-mediated marketing agents. I would be keen to build on your methodology by examining how the disclosure of AI involvement shapes customer experience both in education and broader marketing applications, potentially leading to collaborative projects on AI transparency and consumer behaviour.  With a Master's degree in Marketing Management and seven years of professional experience in data analysis, I am currently working as a Business Intelligence Specialist at ActeroPharma. I am passionate about using advanced analytics to uncover patterns and drive decision-making. Your focus on urban economics and macro-economic studies also resonates with my aspiration to integrate data science with real-world applications.  I would greatly appreciate the opportunity to collaborate with you on research projects or contribute to your ongoing initiatives as a Master/PhD candidate. I have attached my CV for your reference and would be delighted to discuss how my background and skills align with your work.  Thank you for considering my inquiry. I look forward to the possibility of connecting with you.  Best regards, Farnaz Masoumi Master of Marketing Management Central Tehran Azad University farnazmasoumi1988@gmail.com | +98-(933)-099 10 32 """
    professor_reply_body = """Hi Farnaz, Thank you for reaching out and for your interest in our programs. While I am available and open to supervising or co-supervising a student next September, I strongly recommend that you first discuss your eligibility and program fit with the program offices (that is if you haven't already). For our MScB program, details are available here: https://www.dal.ca/study/program-sites/msc-business.html. You may contact Dr. Hélène Deval and Maggie Lapp at mscb@dal.ca. For our PhD program, information can be found here: https://www.dal.ca/faculty/management/programs/phd-programs/phd-management.html. Inquiries can be sent to phdmgmt@dal.ca (the program office will direct you to the appropriate contact). Once you've had some discussions with them and determined the best pathway, we can then talk further about your research objectives and who would be the best fit here at Dal. I hope this helps, and please let me know if you have any questions. Best, Tom Thomas K.B. Koo Assistant Professor FACULTY OF MANAGEMENT | Department of Marketing Kenneth C. Rowe Management Building, Room 5108 DALHOUSIE UNIVERSITY 902.431.5877 ________________________________ From: Farnaz Masoumi <farnazmasoumi1988@gmail.com> Sent: Tuesday, September 30, 2025 12:51 PM To: Tom Koo <tom.kb.koo@dal.ca> Subject: Prospective Student - Farnaz Masoumi CAUTION: The Sender of this email is not from within Dalhousie. Dear Prof. Koo, I hope this message finds you well. My name is Farnaz Masoumi, and I hold a Master's degree in Marketing Management from Central Tehran Azad University. I am writing to express my interest in pursuing a Master/PhD position under your supervision, focusing on consumer behaviour and customer experience. I have been following your research with great interest, particularly your 2024 article titled "The Impact of Generative AI on Syllabus Design and Learning," published in the Journal of Marketing Education. Your paper's insights into students’ preference for AI-generated syllabi—especially its connection to perceptions of objectivity—offer an exciting perspective on how technology can influence educational experiences. With my background in consumer behavior and customer experience, I am eager to explore how similar mechanisms of perceived objectivity might affect consumer trust and engagement in AI-mediated marketing agents. I would be keen to build on your methodology by examining how the disclosure of AI involvement shapes customer experience both in education and broader marketing applications, potentially leading to collaborative projects on AI transparency and consumer behaviour. With a Master's degree in Marketing Management and seven years of professional experience in data analysis, I am currently working as a Business Intelligence Specialist at ActeroPharma. I am passionate about using advanced analytics to uncover patterns and drive decision-making. Your focus on urban economics and macro-economic studies also resonates with my aspiration to integrate data science with real-world applications. I would greatly appreciate the opportunity to collaborate with you on research projects or contribute to your ongoing initiatives as a Master/PhD candidate. I have attached my CV for your reference and would be delighted to discuss how my background and skills align with your work. Thank you for considering my inquiry. I look forward to the possibility of connecting with you. Best regards, Farnaz Masoumi Master of Marketing Management Central Tehran Azad University farnazmasoumi1988@gmail.com<mailto:farnazmasoumi1988@gmail.com> | +98-(933)-099 10 32"""
    main_email_subject = "Prospective Student - Farnaz Masoumi"
    professor_name = "Thomas K. B. Koo"

    tasks = [
        agent.digest(
            funding_request_id=1434,
            professor_reply_body=professor_reply_body,
            main_email_body=main_email_body,
            main_email_subject=main_email_subject,
            professor_name=professor_name,
            regen=True
        )
        for _ in range(10)
    ]
    result = await asyncio.gather(*tasks)
    # result_pd = pd.DataFrame(result).to_dict(orient='list')
    print("Digested Reply:")
    print(json.dumps(result, indent=4))


async def run_benchmark(n_emails: int = 20, n_runs: int = 10):
    """Run benchmark: n_emails x n_runs iterations."""
    print(f"🔄 Fetching {n_emails} random emails...")
    query = """
            SELECT funding_request_id, \
                   professor_reply_body, \
                   main_email_body, \
                   main_email_subject, \
                   professor_name
            FROM funding_emails
            WHERE professor_reply_body IS NOT NULL
              AND professor_reply_body != ''
            ORDER BY RAND()
                LIMIT %s \
            """
    emails = await DB.fetch_all(query, (n_emails,))

    if not emails:
        print("❌ No emails found with professor replies.")
        return []

    print(f"✅ Found {len(emails)} emails. Running {n_runs} iterations each...\n")

    agent = ReplyDigestionAgent()
    results = []

    for i, email in enumerate(emails):
        funding_id = email["funding_request_id"]
        print(f"[{i + 1}/{len(emails)}] Processing funding_request_id={funding_id}...")

        # Run n_runs times for this email
        tasks = [
            agent.digest(
                funding_request_id=funding_id,
                professor_reply_body=email["professor_reply_body"],
                main_email_body=email["main_email_body"],
                main_email_subject=email["main_email_subject"],
                professor_name=email["professor_name"],
                regen=True
            )
            for _ in range(n_runs)
        ]
        run_results = await asyncio.gather(*tasks)

        # Collect metrics for this email
        engagement_labels = [r["engagement_label"] for r in run_results]
        activity_statuses = [r["activity_status"] for r in run_results]
        confidences = [r["confidence"] for r in run_results]

        # Calculate consistency (how often the most common label appears)
        engagement_counter = Counter(engagement_labels)
        most_common_engagement, most_common_count = engagement_counter.most_common(1)[0]
        engagement_consistency = most_common_count / n_runs

        activity_counter = Counter(activity_statuses)
        most_common_activity, activity_common_count = activity_counter.most_common(1)[0]
        activity_consistency = activity_common_count / n_runs

        results.append({
            "funding_request_id": funding_id,
            "engagement_labels": dict(engagement_counter),
            "engagement_majority": most_common_engagement,
            "engagement_consistency": engagement_consistency,
            "activity_statuses": dict(activity_counter),
            "activity_majority": most_common_activity,
            "activity_consistency": activity_consistency,
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "reply_preview": email["professor_reply_body"][:200] + "..."
        })

    # Aggregate metrics
    print("\n" + "=" * 80)
    print("📊 BENCHMARK RESULTS")
    print("=" * 80)

    df = pd.DataFrame(results)

    # Overall consistency
    avg_engagement_consistency = df["engagement_consistency"].mean()
    avg_activity_consistency = df["activity_consistency"].mean()
    avg_confidence = df["avg_confidence"].mean()

    print(f"\n📈 OVERALL METRICS (across {len(emails)} emails, {n_runs} runs each)")
    print(f"   Engagement Label Consistency: {avg_engagement_consistency:.1%}")
    print(f"   Activity Status Consistency:  {avg_activity_consistency:.1%}")
    print(f"   Average Confidence:           {avg_confidence:.2f}")

    # Problematic emails (low consistency)
    low_consistency = df[df["engagement_consistency"] < 0.8]
    print(f"\n⚠️  LOW CONSISTENCY EMAILS (engagement_consistency < 80%): {len(low_consistency)}")

    if len(low_consistency) > 0:
        for _, row in low_consistency.iterrows():
            print(f"\n   funding_request_id={row['funding_request_id']}")
            print(f"   Labels: {row['engagement_labels']}")
            print(f"   Consistency: {row['engagement_consistency']:.0%}")
            print(f"   Preview: {row['reply_preview']}")

    # Label distribution
    print(f"\n📊 MAJORITY LABEL DISTRIBUTION:")
    label_counts = Counter(df["engagement_majority"])
    for label, count in label_counts.most_common():
        print(f"   {label}: {count} ({count / len(df):.0%})")

    # Save detailed results
    output_path = "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Detailed results saved to: {output_path}")

    await DB.close()
    return results

async def main():
    option = input("Select option:\n1. Test single agent\n2. Run benchmark\nEnter choice (1/2): ")
    if option.strip() == "1":
        await agent_test()
    elif option.strip() == "2":
        n_emails = int(input("Enter number of emails to test (default 20): ") or "20")
        n_runs = int(input("Enter number of runs per email (default 5): ") or "5")
        await run_benchmark(n_emails=n_emails, n_runs=n_runs)


if __name__ == "__main__":
    asyncio.run(main())
