
def ext_student_name_subject(subject: str) -> str:
    try:
        return subject.split(" - ")[-1].strip()
    except Exception:
        return ""
