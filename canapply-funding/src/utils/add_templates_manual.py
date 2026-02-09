# src/utils/add_templates_manual.py

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel

from src.db.session import DB


class RespSchema(BaseModel):
    corrected_template: str

class Logic:
    cpd = Path(__file__).parent
    cache_dir = cpd / "cache"

    def __init__(self):
        self.templates_raw = {}

    async def get_funding_templates(self):
        query = "SELECT * FROM funding_templates"
        rows = await DB.fetch_all(query)
        return rows

    async def get_funding_requests(self):
        query = "SELECT * FROM funding_requests"
        rows = await DB.fetch_all(query)
        return rows

    async def get_funding_students_templates(self):
        query = "SELECT * FROM funding_student_templates"
        rows = await DB.fetch_all(query)

        templates = {}
        for row in rows:
            if not templates.get(row["student_id"]):
                templates[row["student_id"]] = {}
            templates[row["student_id"]][row["funding_template_id"]] = row
        return templates

    async def main(self):
        funding_templates = await self.get_funding_templates()
        funding_requests = await self.get_funding_requests()
        funding_students_templates = await self.get_funding_students_templates()

        for row in funding_requests:
            if row["student_template_ids"] is None or len(row["student_template_ids"]) < 4:
                data = {}
                templates = funding_students_templates.get(row["student_id"], {})
                if not templates:
                    print(f"No templates found for student {row['student_id']} in request {row['id']}")
                    continue
                if row["match_status"] == 1:
                    data["main"] = templates[1]["id"]
                elif row["match_status"] == 2:
                    data["main"] = templates[2]["id"]
                if templates.get(3, {}).get("id", 0) != 0:
                    data["reminder1"] = templates.get(3, {}).get("id", 0)
                if templates.get(4, {}).get("id", 0) != 0:
                    data["reminder2"] = templates.get(4, {}).get("id", 0)
                if templates.get(5, {}).get("id", 0) != 0:
                    data["reminder3"] = templates.get(5, {}).get("id", 0)

                update_query = "UPDATE funding_requests SET student_template_ids = %s WHERE id = %s"
                await DB.execute(update_query, (json.dumps(data), row["id"]))
            else:
                print(f"Skipping request {row['id']} with existing student_template_ids")



if __name__ == "__main__":
    logic = Logic()
    asyncio.run(logic.main())
    print("Done.")
