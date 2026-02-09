
import pandas as pd
from llm_client import OpenAIClient, GPT5Nano

from src.db.session import DB
from src.agents.reply_digestion.reply_classification import RespSchema, SYSTEM_PROMPT, USER_PROMPT

client = OpenAIClient(
    model=GPT5Nano,
    cache_backend="pg_redis",
    cache_collection="funding_reply_classification_agent_logs",
)

async def main():
    rows = await DB.fetch_all("select * from funding_emails where professor_replied = 1")

    tasks = []
    for row in rows:
        user_prompt = USER_PROMPT.replace("<<STUDENT_EMAIL_BODY>>", row["main_email_body"])
        user_prompt = user_prompt.replace("<<PROFESSOR_REPLY_BODY>>", row["professor_reply_body"])
        user_prompt = user_prompt.replace("<<PROFESSOR_NAME>>", row["professor_name"])
        student_name = row["main_email_subject"].split("-")[-1].strip()
        user_prompt = user_prompt.replace("<<STUDENT_NAME>>", student_name)

        tasks.append(client.get_response(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format=RespSchema,
            identifier=row["id"],
            body=row
        ))

    results = []
    resps = await client.run_batch(tasks)
    for resp in resps:
        output = resp["output"]
        row = resp["body"]
        usage = resp.get("usage", {})
        results.append({
            "id": row["id"],
            "student_email": row["main_email_body"],
            "student_name": row["main_email_subject"].split("-")[-1].strip(),
            "professor_name": row["professor_name"],
            "professor_reply_body": row["professor_reply_body"],
            "usage": usage,
            **output
        })

    df = pd.DataFrame(results)
    df.to_csv("reply_classification_results.csv", index=False)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())