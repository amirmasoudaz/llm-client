# src/db/queries.py

TABLE_CREATE_FUNDING_REQUESTS = """
CREATE TABLE IF NOT EXISTS funding_requests (
    id                    BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    student_id            BIGINT UNSIGNED NOT NULL,
    professor_id          BIGINT UNSIGNED NOT NULL,
    match_status          TINYINT NOT NULL,

    research_interest     TEXT DEFAULT NULL,
    paper_title           TEXT DEFAULT NULL,
    journal               TEXT DEFAULT NULL,
    year                  INT UNSIGNED DEFAULT NULL,
    research_connection   TEXT DEFAULT NULL,

    attachments           JSON DEFAULT NULL,
    student_template_ids  JSON DEFAULT NULL,

    status                VARCHAR(100) DEFAULT '0',
    ai_status             VARCHAR(255) DEFAULT NULL,
    ai_response           JSON DEFAULT NULL,
    ai_updated_at         TIMESTAMP NULL DEFAULT NULL,

    email_subject         TEXT DEFAULT NULL,
    email_content         TEXT DEFAULT NULL,

    created_at            TIMESTAMP NULL DEFAULT NULL,
    updated_at            TIMESTAMP NULL DEFAULT NULL,
    deleted_at            TIMESTAMP NULL DEFAULT NULL
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;
"""

TABLE_CREATE_FUNDING_EMAILS = """
CREATE TABLE IF NOT EXISTS funding_emails (
    id                      INT AUTO_INCREMENT PRIMARY KEY,
    funding_request_id      INT NOT NULL,
    professor_email         VARCHAR(255) DEFAULT NULL,
    professor_name          VARCHAR(255) DEFAULT NULL,
    main_email_subject      VARCHAR(255) DEFAULT NULL,
    main_email_body         LONGTEXT DEFAULT NULL,
    attachments             JSON DEFAULT NULL,
    gmail_msg_id            VARCHAR(64) DEFAULT NULL,
    gmail_thread_id         VARCHAR(64) DEFAULT NULL,
    main_sent               TINYINT(1) DEFAULT 0,
    main_sent_at            DATETIME DEFAULT NULL,
    professor_replied       TINYINT(1) DEFAULT 0,
    professor_replied_at    DATETIME DEFAULT NULL,
    professor_reply_body    LONGTEXT DEFAULT NULL,
    last_reply_check_at     DATETIME DEFAULT NULL,
    next_reply_check_at     DATETIME DEFAULT NULL,
    reminder_one_sent       TINYINT(1) DEFAULT 0,
    reminder_one_sent_at    DATETIME DEFAULT NULL,
    reminder_one_subject    VARCHAR(255) DEFAULT NULL,
    reminder_one_body       LONGTEXT DEFAULT NULL,
    reminder_two_sent       TINYINT(1) DEFAULT 0,
    reminder_two_sent_at    DATETIME DEFAULT NULL,
    reminder_two_subject    VARCHAR(255) DEFAULT NULL,
    reminder_two_body       LONGTEXT DEFAULT NULL,
    reminder_three_sent     TINYINT(1) DEFAULT 0,
    reminder_three_sent_at  DATETIME DEFAULT NULL,
    reminder_three_subject  VARCHAR(255) DEFAULT NULL,
    reminder_three_body     LONGTEXT DEFAULT NULL,
    no_more_reminders       TINYINT(1) DEFAULT 0,
    no_more_checks          TINYINT(1) DEFAULT 0,
    created_at              DATETIME DEFAULT NULL,
    updated_at              DATETIME DEFAULT NULL,
    UNIQUE KEY uq_funding_request (funding_request_id),
    KEY idx_funding_emails_status (main_sent, professor_replied, no_more_reminders, no_more_checks),
    KEY idx_funding_emails_next_check (next_reply_check_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
"""

TABLE_CREATE_FUNDING_STUDENT_TEMPLATES = """
CREATE TABLE IF NOT EXISTS funding_student_templates (
    id                      BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    student_id              BIGINT UNSIGNED NOT NULL,
    funding_template_id     BIGINT UNSIGNED NOT NULL,
    content                 LONGTEXT NOT NULL,
    formatted_content       LONGTEXT DEFAULT NULL,
    subject                 VARCHAR(255) NOT NULL,
    formatted_subject       VARCHAR(255) DEFAULT NULL,
    variables               JSON NOT NULL,
    active                  TINYINT(1) NOT NULL DEFAULT 1,
    created_at              TIMESTAMP NULL DEFAULT NULL,
    updated_at              TIMESTAMP NULL DEFAULT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY funding_student_templates_student_id_funding_template_id_unique (student_id, funding_template_id),
    KEY funding_student_templates_funding_template_id_foreign (funding_template_id),
    CONSTRAINT funding_student_templates_funding_template_id_foreign
        FOREIGN KEY (funding_template_id) REFERENCES funding_templates(id)
        ON DELETE CASCADE,
    CONSTRAINT funding_student_templates_student_id_foreign
        FOREIGN KEY (student_id) REFERENCES students(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

TABLE_CREATE_PROFESSOR_TAGS = """
CREATE TABLE IF NOT EXISTS professor_tags (
    prof_hash       BINARY(32) NOT NULL PRIMARY KEY,
    tags_json       JSON NOT NULL,
    confidence      FLOAT DEFAULT NULL,
    sources_used    JSON DEFAULT NULL,
    notes           TEXT DEFAULT NULL,
    generated_at    TIMESTAMP NOT NULL 
                        DEFAULT CURRENT_TIMESTAMP 
                        ON UPDATE CURRENT_TIMESTAMP,
    KEY idx_prof_hash (prof_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
"""

TABLE_CREATE_OAUTH_CLIENTS = """
CREATE TABLE IF NOT EXISTS oauth_clients (
    id              INT(11) AUTO_INCREMENT PRIMARY KEY,
    user_id         BIGINT(20) UNSIGNED NOT NULL,
    client_id       VARCHAR(255) NOT NULL,
    client_secret   VARCHAR(255) NOT NULL,
    scopes          JSON NOT NULL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_oauth_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

TABLE_CREATE_OAUTH_SESSIONS = """
CREATE TABLE IF NOT EXISTS oauth_sessions (
    state           VARCHAR(255) NOT NULL PRIMARY KEY,
    user_id         BIGINT(20) UNSIGNED NOT NULL,
    code_verifier   VARCHAR(255) NOT NULL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

TABLE_CREATE_FUNDING_CREDENTIALS = """
CREATE TABLE IF NOT EXISTS funding_credentials (
    id            INT(11) AUTO_INCREMENT PRIMARY KEY,
    user_id       INT(10) UNSIGNED NOT NULL,
    token_blob    JSON NOT NULL,
    reminders_on  TINYINT(1) NOT NULL DEFAULT 1,
    last_refresh  DATETIME DEFAULT NULL,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_user_id (user_id),
    CONSTRAINT fk_credentials_student
        FOREIGN KEY (user_id) REFERENCES students(id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

TABLE_CREATE_FUNDING_INSTITUTES = """
CREATE TABLE IF NOT EXISTS funding_institutes (
    id                INT(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    institution_name  VARCHAR(255) NOT NULL,
    department_name   VARCHAR(255) NOT NULL,
    institution_url   VARCHAR(1024) DEFAULT NULL,
    logo_address      VARCHAR(1024) DEFAULT NULL,
    city              VARCHAR(255) DEFAULT NULL,
    province          VARCHAR(255) DEFAULT NULL,
    country           VARCHAR(255) DEFAULT NULL,
    is_active         TINYINT(1) NOT NULL DEFAULT 1,
    UNIQUE KEY uniq_inst_dept (institution_name, department_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
"""

TABLE_CREATE_FUNDING_PROFESSORS = """
CREATE TABLE IF NOT EXISTS funding_professors (
    prof_hash           BINARY(32) NOT NULL,
    id                  INT(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    full_name           VARCHAR(512) NOT NULL,
    first_name          VARCHAR(512) NOT NULL,
    middle_name         VARCHAR(512) DEFAULT NULL,
    last_name           VARCHAR(512) NOT NULL,
    occupation          VARCHAR(512) DEFAULT NULL,
    department          VARCHAR(255) NOT NULL,
    email_address       VARCHAR(320) NOT NULL,
    url                 VARCHAR(1024) DEFAULT NULL,
    funding_institute_id INT(10) UNSIGNED NOT NULL,
    other_contact_info  JSON DEFAULT NULL,
    research_areas      JSON NOT NULL,
    credentials         TEXT DEFAULT NULL,
    area_of_expertise   JSON DEFAULT NULL,
    categories          JSON NOT NULL,
    others              JSON DEFAULT NULL,
    is_active           TINYINT(1) NOT NULL DEFAULT 1,
    UNIQUE KEY uniq_prof_hash (prof_hash),
    KEY fk_prof_inst (funding_institute_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
"""

TABLE_CREATE_FUNDING_REPLIES = """
CREATE TABLE funding_replies (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    funding_request_id BIGINT UNSIGNED NOT NULL,
    reply_body_raw LONGTEXT NOT NULL,
    reply_body_cleaned LONGTEXT NOT NULL,    
    needs_human_review TINYINT(1) DEFAULT 0,
    is_auto_generated TINYINT(1) DEFAULT 0,
    auto_generated_type VARCHAR(50) DEFAULT 'NONE',
    engagement_label VARCHAR(100) DEFAULT NULL,
    engagement_bool TINYINT(1) DEFAULT NULL,
    activity_status VARCHAR(100) DEFAULT NULL,
    activity_bool TINYINT(1) DEFAULT NULL,    
    short_rationale TEXT DEFAULT NULL,
    key_phrases JSON DEFAULT NULL,    
    confidence FLOAT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_funding_request_id (funding_request_id)
);
"""

TABLE_CREATE_STUDENT_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS student_documents (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    student_id BIGINT UNSIGNED NOT NULL,
    title VARCHAR(255) NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    document_type VARCHAR(255) NOT NULL,
    status VARCHAR(255) NOT NULL,
    document_hash VARCHAR(255) NOT NULL,
    raw_content TEXT DEFAULT NULL,
    json_content JSON DEFAULT NULL,
    parser_version VARCHAR(255) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT student_documents_student_id_foreign FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

SELECT_REQUEST = "SELECT * FROM funding_requests WHERE id=%s"
SELECT_PROFESSOR = "SELECT * FROM funding_professors WHERE id=%s"
# SELECT_DRAFT = "SELECT * FROM funding_emails WHERE funding_request_id=%s"
SELECT_REVIEW_INPUTS = """
SELECT
    fr.student_id AS student_id,
    fr.professor_id AS professor_id,
    fr.match_status AS match_status,
    fr.attachments AS email_attachments,
    fr.research_interest AS research_interest,
    fr.paper_title AS paper_title,
    fr.journal AS paper_journal,
    fr.year AS paper_year,
    fr.research_connection AS research_connection,
    fr.student_template_ids AS student_template_ids,
    fp.full_name AS professor_full_name,
    fp.last_name AS professor_last_name,
    fp.email_address AS professor_email
FROM funding_requests fr
JOIN funding_professors fp ON fr.professor_id = fp.id
WHERE fr.id=%s
"""

SELECT_SEND_INPUTS = """
SELECT 
    fe.id AS funding_email_id,
    fe.main_sent AS main_email_sent,
    fp.email_address AS professor_email,
    fr.email_subject AS email_subject,
    fr.email_content AS email_body,
    fr.attachments AS email_attachments,
    fr.student_id AS student_id,
FROM funding_requests fr
JOIN funding_professors fp ON fr.professor_id = fp.id
JOIN funding_emails fe ON fr.id = fe.funding_request_id
WHERE fr.id=%s
"""

UPSERT_DRAFT = """
INSERT INTO funding_emails
    (funding_request_id, student_id, professor_email, professor_name,
     main_email_subject, main_email_body, attachments, created_at, updated_at)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON DUPLICATE KEY UPDATE
    main_email_subject  = VALUES(main_email_subject),
    main_email_body     = VALUES(main_email_body),
    attachments         = VALUES(attachments),
    updated_at          = VALUES(updated_at)
"""

UPDATE_REQUEST = """
UPDATE funding_requests
SET email_subject=%s, email_content=%s, updated_at=%s
WHERE id=%s
"""

MARK_SENT = """
UPDATE funding_emails
SET main_sent=1, gmail_msg_id=%s, gmail_thread_id=%s, main_sent_at=%s, updated_at=%s,
    main_email_subject=%s, main_email_body=%s
WHERE id=%s
"""

SELECT_PENDING_FOR_CHECK = """
SELECT
    fe.*,
    fr.student_id,
    fr.student_template_ids
FROM funding_emails fe
JOIN funding_requests fr ON fr.id = fe.funding_request_id
JOIN funding_credentials fc ON fc.user_id = fr.student_id
WHERE fe.main_sent = 1
  AND fe.professor_replied = 0
  AND fe.no_more_checks = 0
  AND (fe.next_reply_check_at IS NULL OR fe.next_reply_check_at <= NOW())
"""

GET_EMAIL_CONTENT_EMPTY_IDS = """
SELECT id FROM funding_requests
WHERE email_content = '' OR email_content IS NULL
"""

MARK_PROFESSOR_REPLIED = """
UPDATE funding_emails
SET professor_replied = 1,
    professor_replied_at = %s,
    professor_reply_body = %s,
    updated_at = %s,
    last_reply_check_at = %s,
    next_reply_check_at = NULL,
    no_more_checks = 1,
    no_more_reminders = 1
WHERE id = %s
"""

MARK_REMINDER_ONE_SENT = """
UPDATE funding_emails
SET reminder_one_sent = 1,
    reminder_one_subject = %s,
    reminder_one_body = %s,
    reminder_one_sent_at = %s,
    updated_at = %s
WHERE id = %s
"""

MARK_REMINDER_TWO_SENT = """
UPDATE funding_emails
SET reminder_two_sent = 1,
    reminder_two_subject = %s,
    reminder_two_body = %s,
    reminder_two_sent_at = %s,
    updated_at = %s
WHERE id = %s
"""

MARK_REMINDER_THREE_SENT = """
UPDATE funding_emails
SET reminder_three_sent = 1,
    reminder_three_subject = %s,
    reminder_three_body = %s,
    reminder_three_sent_at = %s,
    no_more_reminders = 1,
    updated_at = %s
WHERE id = %s
"""

MARK_REPLY_CHECKED = """
UPDATE funding_emails
SET last_reply_check_at = %s,
    next_reply_check_at = %s,
    updated_at = %s
WHERE id = %s
"""

MARK_NO_MORE_CHECKS = """
UPDATE funding_emails
SET no_more_checks = 1,
    updated_at = %s
WHERE id = %s
"""

SELECT_GMAIL_TOKEN = """
SELECT token_blob FROM funding_credentials WHERE user_id=%s
"""

UPDATE_GMAIL_TOKEN = """
UPDATE funding_credentials
SET token_blob=%s, last_refresh=UTC_TIMESTAMP()
WHERE user_id=%s
"""

SELECT_UNIQUE_INSTITUTES = """
SELECT DISTINCT id, institution_name
FROM funding_institutes
ORDER BY institution_name
"""

SELECT_ALL_PROFESSORS = f"""
SELECT
  p.prof_hash,
  p.full_name,
  p.occupation,
  p.department,
  p.url,
  p.research_areas,
  p.credentials,
  p.area_of_expertise,
  p.others,
  i.institution_name AS institute,
  i.country AS country
FROM funding_professors p
JOIN funding_institutes i ON p.funding_institute_id = i.id
WHERE p.is_active = 1
"""

SELECT_ALL_PROF_HASHES_FROM_TAGS = """
SELECT DISTINCT p.prof_hash
FROM professor_tags AS p
INNER JOIN funding_professors AS f 
    ON p.prof_hash = f.prof_hash
INNER JOIN funding_institutes AS i 
    ON f.funding_institute_id = i.id
WHERE f.is_active = 1
  AND i.is_active = 1
  AND f.email_address IS NOT NULL
  AND f.email_address <> ''
"""
SELECT_ALL_PROF_HASHES_FROM_PROFS = "SELECT prof_hash FROM funding_professors WHERE is_active=1"

UPSERT_PROF_TAGS = f"""
INSERT INTO professor_tags (prof_hash, tags_json, confidence, sources_used, notes)
VALUES (%s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
  tags_json = VALUES(tags_json),
  confidence = VALUES(confidence),
  sources_used = VALUES(sources_used),
  notes = VALUES(notes);
"""

INSERT_GMAIL_CREDENTIALS = """
INSERT INTO funding_credentials (user_id, token_blob)
VALUES (%s, %s)
ON DUPLICATE KEY UPDATE token_blob = VALUES(token_blob)
"""

TRUNCATE_GMAIL_CREDENTIALS = """
TRUNCATE TABLE funding_credentials
"""
