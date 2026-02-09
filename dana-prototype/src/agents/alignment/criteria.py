# src/agents/alignment/criteria.py

ALIGNMENT_CRITERIA = {
    "alignment_analysis": [
        {
            "category_id": "RA",
            "category_name": "Research Alignment",
            "questions": [
                {
                    "id": "ALN_RA_01",
                    "question": "Do the user's research interests significantly overlap with the professor's research areas?",
                    "weight": 10,
                    "required": True,
                    "instructions": "1. Identify the professor's primary research areas from PROFESSOR_PROFILE. 2. Analyze the user's stated research interests from USER_PROFILE and past research from USER_RESUME. 3. Yes: If there is substantial thematic overlap (≥60%). Intensity: Correlates with depth and breadth of overlap. No: If interests are in different domains. Intensity: High if completely unrelated."
                },
                {
                    "id": "ALN_RA_02",
                    "question": "Has the user demonstrated commitment to research topics relevant to the professor's work?",
                    "weight": 8,
                    "required": True,
                    "instructions": "Look for evidence in USER_RESUME: publications, projects, coursework, or sustained focus in related areas. Yes if ≥2 significant indicators of commitment. Intensity scales with quantity and quality of evidence."
                },
                {
                    "id": "ALN_RA_03",
                    "question": "Does the user's research vision or goals align with potential directions in the professor's lab?",
                    "weight": 7,
                    "required": False,
                    "instructions": "Analyze USER_PROFILE for research goals and compare with PROFESSOR_PROFILE research directions. Yes if future goals are compatible with lab's trajectory. Intensity based on alignment strength."
                }
            ]
        },
        {
            "category_id": "TS",
            "category_name": "Technical Skills",
            "questions": [
                {
                    "id": "ALN_TS_01",
                    "question": "Does the user possess the core technical skills needed for the professor's research?",
                    "weight": 9,
                    "required": True,
                    "instructions": "1. Identify technical skills/methodologies mentioned in PROFESSOR_PROFILE (e.g., machine learning, molecular biology techniques, statistical analysis). 2. Check USER_RESUME for evidence of these skills in coursework, projects, or experience. 3. Yes: If user has ≥70% of core skills. Intensity: Proportional to skill coverage. No: If major skills are missing. Intensity: Based on gap severity."
                },
                {
                    "id": "ALN_TS_02",
                    "question": "Has the user demonstrated practical application of relevant technical skills?",
                    "weight": 7,
                    "required": False,
                    "instructions": "Look for projects, publications, or work experience showing hands-on use of relevant skills. Yes if evidence of practical application exists. Intensity based on depth and recency of application."
                },
                {
                    "id": "ALN_TS_03",
                    "question": "Does the user show proficiency in tools or platforms commonly used by the professor's lab?",
                    "weight": 5,
                    "required": False,
                    "instructions": "Check for specific tools, software, or platforms mentioned in PROFESSOR_PROFILE. Yes if user has experience with similar tools. Intensity correlates with exact matches vs. transferable skills."
                }
            ]
        },
        {
            "category_id": "AB",
            "category_name": "Academic Background",
            "questions": [
                {
                    "id": "ALN_AB_01",
                    "question": "Does the user's educational background align with the professor's field?",
                    "weight": 8,
                    "required": True,
                    "instructions": "1. Identify the professor's field from PROFESSOR_PROFILE. 2. Analyze USER_RESUME education section for degree(s) and major(s). 3. Yes: If degree is in the same or closely related field. Intensity: 1.0 for direct match, lower for related fields. No: If field is unrelated. Intensity: Based on distance between fields."
                },
                {
                    "id": "ALN_AB_02",
                    "question": "Has the user completed relevant coursework or training for this research area?",
                    "weight": 6,
                    "required": False,
                    "instructions": "Look for courses, certifications, or training programs in USER_RESUME relevant to professor's research. Yes if relevant coursework exists. Intensity based on quantity and level of coursework."
                },
                {
                    "id": "ALN_AB_03",
                    "question": "Is the user's academic level appropriate for the professor's typical students?",
                    "weight": 7,
                    "required": True,
                    "instructions": "Infer from PROFESSOR_PROFILE what level of students the professor typically works with (PhD, Masters, undergrad). Compare with user's current academic status from USER_PROFILE. Yes if levels match. Intensity: High for exact match, moderate for adjacent levels."
                }
            ]
        },
        {
            "category_id": "EM",
            "category_name": "Experience Match",
            "questions": [
                {
                    "id": "ALN_EM_01",
                    "question": "Does the user have research experience relevant to the professor's work?",
                    "weight": 8,
                    "required": False,
                    "instructions": "Analyze USER_RESUME for research positions, lab work, or research projects. Yes if relevant research experience exists. Intensity correlates with amount and relevance of experience."
                },
                {
                    "id": "ALN_EM_02",
                    "question": "Has the user worked on projects or problems similar to those in the professor's lab?",
                    "weight": 7,
                    "required": False,
                    "instructions": "Compare project descriptions in USER_RESUME with research topics in PROFESSOR_PROFILE. Yes if similar problems or methodologies. Intensity based on similarity and scope."
                }
            ]
        },
        {
            "category_id": "PI",
            "category_name": "Publications & Impact",
            "questions": [
                {
                    "id": "ALN_PI_01",
                    "question": "Does the user have publications, presentations, or academic output relevant to the professor's field?",
                    "weight": 6,
                    "required": False,
                    "instructions": "Look for publications, conference presentations, posters, or other research output in USER_RESUME. Yes if relevant output exists. Intensity based on quantity, quality, and relevance. If user is early-stage with no output, answer Yes with low intensity (0.2-0.3) to avoid penalizing."
                },
                {
                    "id": "ALN_PI_02",
                    "question": "Does the user demonstrate research productivity compatible with the professor's expectations?",
                    "weight": 5,
                    "required": False,
                    "instructions": "Consider the user's career stage and output. For early-stage researchers, course projects and independent studies count. Yes if productivity is reasonable for career stage. Intensity based on output quality and quantity relative to stage."
                }
            ]
        }
    ]
}
