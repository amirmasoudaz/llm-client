from __future__ import annotations


SELECT_FUNDING_THREAD_CONTEXT = """
SELECT
    s.id AS user_id,
    fr.id AS request_id,
    fe.id AS email_id,
    fp.id AS request_professor_id,
    s.first_name AS user_first_name,
    s.last_name AS user_last_name,
    s.email AS user_email_address,
    s.mobile_number AS user_phone_number,
    s.date_of_birth AS user_date_of_birth,
    s.gender AS user_gender,
    s.country_of_citizenship as user_country_of_citizenship,
    fr.created_at AS request_creation_datetime,
    fr.match_status AS request_match_status,
    fr.research_interest AS request_research_interest,
    fr.paper_title AS request_paper_title,
    fr.journal AS request_journal,
    fr.`year` AS request_year,
    fr.research_connection AS request_research_connection,
    fr.attachments AS request_attachments,
    fr.student_template_ids AS request_student_template_ids,
    fr.`status` AS request_status,
    fr.email_subject AS request_email_subject,
    fr.email_content AS request_email_body,
    fe.main_sent AS main_email_sent_status,
    fe.main_sent_at AS main_email_sent_datetime,
    fe.main_email_subject AS main_email_subject,
    fe.main_email_body AS main_email_body,
    fe.reminder_one_sent AS reminder_one_sent_status,
    fe.reminder_one_sent_at AS reminder_one_sent_datetime,
    fe.reminder_one_body AS reminder_one_body,
    fe.reminder_two_sent AS reminder_two_sent_status,
    fe.reminder_two_sent_at AS reminder_two_sent_datetime,
    fe.reminder_two_body AS reminder_two_body,
    fe.reminder_three_sent AS reminder_three_sent_status,
    fe.reminder_three_sent_at AS reminder_three_sent_datetime,
    fe.reminder_three_body AS reminder_three_body,
    fe.professor_replied AS professor_reply_status,
    fe.professor_replied_at AS professor_reply_datetime,
    frep.reply_body_cleaned AS professor_reply_body,
    frep.engagement_label AS professor_reply_engagement_label,
    frep.activity_status AS professor_reply_activity_status,
    frep.short_rationale AS professor_reply_short_rationale,
    frep.key_phrases AS professor_reply_key_phrases,
    frep.auto_generated_type AS professor_reply_auto_generated_type,
    fp.first_name AS professor_first_name,
    fp.last_name AS professor_last_name,
    fp.full_name AS professor_full_name,
    fp.occupation AS professor_occupation,
    fp.research_areas AS professor_research_areas,
    fp.credentials AS professor_credentials,
    fp.area_of_expertise AS professor_area_of_expertise,
    fp.categories AS professor_categories,
    fp.department AS professor_department,
    fp.email_address AS professor_email_address,
    fp.url AS professor_url,
    fp.others AS professor_others,
    fi.institution_name AS institute_name,
    fi.department_name AS department_name,
    fi.country AS institute_country,
    m.value AS user_onboarding_data
FROM funding_requests fr
JOIN students s ON s.id = fr.student_id
JOIN funding_professors fp ON fp.id = fr.professor_id
JOIN funding_institutes fi ON fi.id = fp.funding_institute_id
LEFT JOIN metas m ON m.metable_id = fr.student_id AND m.`key` = 'funding_template_initial_data'
LEFT JOIN funding_emails fe ON fr.id = fe.funding_request_id
LEFT JOIN funding_replies frep ON fr.id = frep.funding_request_id
WHERE fr.id = %s
LIMIT 1;
"""

