# src/agents/resume/latex/template.py

"""
LaTeX template rendering for AcademicCV.
Matches the styling of examples/example_cv exactly.
"""

import re
from typing import List
from agents.resume.resume_generation import PublicationItem


CLS_CONTENT = """
\ProvidesClass{resume}[2010/07/10 v0.9 Resume class]

\LoadClass[11pt,letterpaper]{article}

\usepackage[parfill]{parskip}
\usepackage{array}
\usepackage{ifthen}
\usepackage{hyperref}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,
    urlcolor=cyan,
}

\pagestyle{empty}

\def \name#1{\def\@name{#1}}
\def \@name {}

\def \addressSep {$\diamond$}

\let \@addressone \relax
\let \@addresstwo \relax
\let \@addressthree \relax

\def \address #1{
  \@ifundefined{@addresstwo}{
    \def \@addresstwo {#1}
  }{
  \@ifundefined{@addressthree}{
  \def \@addressthree {#1}
  }{
     \def \@addressone {#1}
  }}
}

\def \printaddress #1{
  \begingroup
    \def \\ {\addressSep\ }
    \centerline{#1}
  \endgroup
  \par
  \addressskip
}

\def \printname {
  \begingroup
    \hfil{\MakeUppercase{\namesize\bf \@name}}\hfil
    \nameskip\break
  \endgroup
}

\let\ori@document=\document
\renewcommand{\document}{
  \ori@document
  \printname
  \@ifundefined{@addressone}{}{
    \printaddress{\@addressone}}
  \@ifundefined{@addresstwo}{}{
    \printaddress{\@addresstwo}}
  \@ifundefined{@addressthree}{}{
    \printaddress{\@addressthree}}
}

\newenvironment{rSection}[1]{
  \sectionskip
  \MakeUppercase{\bf #1}
  \sectionlineskip
  \hrule
  \begin{list}{}{
    \setlength{\leftmargin}{0em}
  }
  \item[]
}{
  \end{list}
}

\newenvironment{rSubsection}[4]{
 {\bf #1} \hfill {#2}
 \ifthenelse{\equal{#3}{}}{}{
  \\
  {\em #3} \hfill {\em #4}
  }\smallskip
  \begin{list}{$\cdot$}{\leftmargin=0em}
   \itemsep -0.5em \vspace{-0.5em}
  }{
  \end{list}
  \vspace{0.5em}
}

\def\namesize{\LARGE}
\def\addressskip{\smallskip}
\def\sectionlineskip{\medskip}
\def\nameskip{\medskip}
\def\sectionskip{\medskip}
"""


def _make_cite_key(pub: PublicationItem) -> str:
    """Generate a stable BibTeX cite key from publication info."""
    # Extract first author's last name
    first_author = pub.authors[0] if pub.authors else "unknown"
    # Get last name (handle "Last, First" or "First Last" formats)
    if "," in first_author:
        last_name = first_author.split(",")[0].strip()
    else:
        parts = first_author.split()
        last_name = parts[-1] if parts else "unknown"

    # Clean the name for BibTeX key
    last_name = re.sub(r'[^a-zA-Z]', '', last_name).lower()

    # Get year
    year = pub.year if pub.year else "0000"

    # Get first significant word from title
    title_words = re.sub(r'[^a-zA-Z\s]', '', pub.title or "untitled").lower().split()
    stopwords = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "with"}
    title_word = next((w for w in title_words if w not in stopwords), "")[:10]

    return f"{last_name}{year}{title_word}"


def _escape_bibtex(s: str) -> str:
    """Escape special characters for BibTeX."""
    if not s:
        return ""
    # Wrap in braces to preserve capitalization
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("_", r"\_")
    s = s.replace("#", r"\#")
    return s


def _publication_type_to_bibtex(pub: PublicationItem) -> str:
    """Map PublicationItem type to BibTeX entry type."""
    type_map = {
        "Journal article": "article",
        "Conference paper": "inproceedings",
        "Book chapter": "incollection",
        "Case report": "misc",
        "Thesis": "phdthesis",
        "Other": "misc",
    }
    # Also check status
    if pub.status == "Conference":
        return "inproceedings"
    if pub.status == "Preprint":
        return "misc"
    return type_map.get(pub.publication_type, "misc")


def generate_bibtex(
        publications: List[PublicationItem],
        cv_owner_name: str,
) -> tuple[str, dict[str, str]]:
    """
    Generate BibTeX content from a list of PublicationItem.

    Args:
        publications: List of PublicationItem from AcademicCV
        cv_owner_name: Name of CV owner for author highlighting annotation

    Returns:
        Tuple of (bibtex_content, cite_key_mapping)
        cite_key_mapping maps original title to cite key for reference
    """
    entries = []
    cite_keys = {}

    for idx, pub in enumerate(publications):
        cite_key = _make_cite_key(pub)
        cite_keys[pub.title] = cite_key

        entry_type = _publication_type_to_bibtex(pub)

        # Build author string with highlight annotation
        authors = []
        owner_patterns = cv_owner_name.lower().split()
        for author in pub.authors:
            author_lower = author.lower()
            # Check if this author is the CV owner
            is_owner = all(p in author_lower for p in owner_patterns)
            if is_owner:
                # Add annotation for highlighting
                authors.append(author + "=highlight")
            else:
                authors.append(author)
        author_str = " and ".join(authors)

        # Build the entry
        lines = [
            f"@{entry_type}{{{cite_key},",
            f"  title   = {{{_escape_bibtex(pub.title)}}},",
            f"  author  = {{{author_str}}},"
        ]

        if pub.venue:
            if entry_type == "article":
                lines.append(f"  journal = {{{_escape_bibtex(pub.venue)}}},")
            elif entry_type == "inproceedings":
                lines.append(f"  booktitle = {{{_escape_bibtex(pub.venue)}}},")
            else:
                lines.append(f"  note    = {{{_escape_bibtex(pub.venue)}}},")

        if pub.year:
            lines.append(f"  year    = {{{pub.year}}},")

        if pub.doi_or_url:
            if pub.doi_or_url.startswith("10.") or "doi.org" in pub.doi_or_url:
                lines.append(f"  doi     = {{{pub.doi_or_url}}},")
            else:
                lines.append(f"  url     = {{{pub.doi_or_url}}},")

        if pub.notes:
            lines.append(f"  note    = {{{_escape_bibtex(pub.notes)}}},")

        # Remove trailing comma from last line
        if lines[-1].endswith(","):
            lines[-1] = lines[-1][:-1]

        lines.append("}")
        entries.append("\n".join(lines))

    bibtex_content = "\n\n".join(entries)
    return bibtex_content, cite_keys


class CVLatexRenderer:
    """Renders AcademicCV JSON to LaTeX matching example_cv styling."""
    
    # Unicode to LaTeX mapping for special characters
    _UNICODE_LATEX_MAP = {
        "×": "\\times",
        "–": "--",
        "—": "---",
        """: '"',
        """: '"',
        "'": "'",
        "…": "...",
    }
    
    def __init__(self, cv: dict):
        """
        Initialize renderer with CV data.
        
        Args:
            cv: AcademicCV dict or JSON
        """
        self.cv = cv
        self.basics = cv.get("basics", {})
        self.sections = cv.get("sections", {})
        self.cite_keys = {}  # Populated by _generate_bibtex
        
    def _normalize_unicode(self, s: str) -> str:
        """Convert Unicode characters to LaTeX equivalents."""
        if not s:
            return ""
        for k, v in self._UNICODE_LATEX_MAP.items():
            if k in s:
                s = s.replace(k, v)
        return s
    
    def _escape_latex(self, s: str) -> str:
        """Escape LaTeX special characters."""
        if not s:
            return ""
        s = self._normalize_unicode(s)
        # Order matters: backslash first
        s = re.sub(r"(?<!\\)%", r"\\%", s)
        s = re.sub(r"(?<!\\)#", r"\\#", s)
        s = re.sub(r"(?<!\\)&", r"\\&", s)
        s = re.sub(r"(?<!\\)\$", r"\\$", s)
        s = re.sub(r"(?<!\\)_", r"\\_", s)
        s = re.sub(r"(?<!\\)\{", r"\\{", s)
        s = re.sub(r"(?<!\\)\}", r"\\}", s)
        s = re.sub(r"(?<!\\)\^", r"\\^{}", s)
        s = re.sub(r"(?<!\\)~", r"\\~{}", s)
        return s
    
    def fmt(self, s: str) -> str:
        """Format string for LaTeX output."""
        return self._escape_latex(s or "")

    @staticmethod
    def _format_date_range(dr: dict) -> str:
        """Format date range for display."""
        if not dr:
            return ""
        start = dr.get("start", "")
        end = dr.get("end", "")
        
        # Format: extract year only for display
        start_year = start[:4] if start else ""
        end_year = end[:4] if end else "Present"
        
        if start_year and end_year:
            return f"{start_year}--{end_year}"
        elif start_year:
            return f"{start_year}--Present"
        return ""
    
    def _format_location(self, loc: dict) -> str:
        """Format location for display."""
        if not loc:
            return ""
        city = loc.get("city", "")
        country = loc.get("country", "")
        if city and country:
            return f"{self.fmt(city)}, {self.fmt(country)}"
        return self.fmt(city or country)
    
    def render(self) -> tuple[str, str, str]:
        """
        Render CV to LaTeX and BibTeX.
        
        Returns:
            Tuple of (tex_content, bib_content)
        """
        # Generate BibTeX first to get cite keys
        bib_content = self._generate_bibtex()
        
        # Build LaTeX document
        tex_parts = [
            self._render_preamble(),
            self._render_header(),
            self._render_profile(),
            self._render_education(),
            self._render_research_experience(),
            self._render_research_interests(),
            self._render_professional_experience(),
            self._render_coursework(),
            self._render_publications(),
            self._render_workshops(),
            self._render_skills(),
            self._render_memberships(),
            self._render_languages(),
            self._render_references(),
            self._render_closing(),
        ]
        
        tex_content = "\n".join(p for p in tex_parts if p)
        return tex_content, bib_content, CLS_CONTENT
    
    def _generate_bibtex(self) -> str:
        """Generate BibTeX for publications."""
        publications = self.sections.get("publications", [])
        if not publications:
            return ""
        
        # Convert dicts to proper format if needed
        cv_owner = self.basics.get("full_name", "")
        
        # Build publication items
        pub_items = []
        for pub in publications:
            if isinstance(pub, dict):
                pub_items.append(PublicationItem(**pub))
            else:
                pub_items.append(pub)
        
        if not pub_items:
            return ""
            
        bib_content, self.cite_keys = generate_bibtex(pub_items, cv_owner)
        return bib_content
    
    def _render_preamble(self) -> str:
        """Render LaTeX document preamble."""
        has_bib = bool(self.sections.get("publications"))
        
        preamble = r"""\documentclass[11pt,letterpaper]{article}

\usepackage[margin=0.6in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{parskip}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{array,tabularx,booktabs}
\usepackage{fontawesome}

\definecolor{accent}{HTML}{1F4E79}
\hypersetup{colorlinks=true,urlcolor=accent,linkcolor=accent,citecolor=accent}
"""
        
        if has_bib:
            preamble += r"""
\usepackage[maxbibnames=99,maxcitenames=1,style=ieee]{biblatex}
\makeatletter
\DeclareCiteCommand{\fullcitation}
  {\defcounter{maxnames}{\blx@maxbibnames}\usebibmacro{prenote}}
  {\usedriver{\DeclareNameAlias{sortname}{default}}{\thefield{entrytype}}}
  {\multicitedelim}
  {\usebibmacro{postnote}}
\makeatother
\renewcommand*{\mkbibnamegiven}[1]{\ifitemannotation{highlight}{\textbf{#1}}{#1}}
\renewcommand*{\mkbibnamefamily}[1]{\ifitemannotation{highlight}{\textbf{#1}}{#1}}
\addbibresource{pubs.bib}
"""
        
        preamble += r"""
\setlist{noitemsep,topsep=2pt,leftmargin=*}

\newcommand{\cvsection}[1]{\vspace{0.75\baselineskip}{\large\bfseries\textcolor{accent}{#1}}\par
\vspace{0.25\baselineskip}\hrule\vspace{0.2\baselineskip}}
\newcommand{\entry}[2]{\noindent\textbf{#1}\hfill #2\par}
\newcommand{\subentry}[2]{\noindent\emph{#1}\hfill \emph{#2}\par}
"""
        return preamble
    
    def _render_header(self) -> str:
        """Render CV header with name and contact info."""
        name = self.fmt(self.basics.get("full_name", ""))
        title = self.fmt(self.basics.get("current_title", ""))
        email = self.basics.get("email", "")
        phone = self.fmt(self.basics.get("phone", ""))
        linkedin = self.basics.get("linkedin", "")
        location = self._format_location(self.basics.get("location", {}))
        
        # Build contact rows
        contact_rows = []
        if email:
            contact_rows.append(rf"    \faEnvelope\ & \href{{mailto:{email}}}{{{self.fmt(email)}}}\\")
        if phone:
            contact_rows.append(rf"    \faPhone\    & {phone}\\")
        if linkedin:
            contact_rows.append(rf"    \faLinkedin\ & \href{{{linkedin}}}{{LinkedIn}}\\")
        if location:
            contact_rows.append(rf"    \faMapMarker & {location}\\")
        
        contact_table = "\n".join(contact_rows)
        
        header = rf"""
\begin{{document}}
\noindent
\begin{{minipage}}[t]{{0.62\textwidth}}
  {{\bfseries\fontsize{{26}}{{26}}\selectfont \textcolor{{accent}}{{{name}}}}}\par
  {title}
\end{{minipage}}%
\hfill
\begin{{minipage}}[t]{{0.35\textwidth}}
  \raggedleft
  \renewcommand{{\arraystretch}}{{1.2}}
  \begin{{tabular}}{{@{{}}r l@{{}}}}
{contact_table}
  \end{{tabular}}
\end{{minipage}}
"""
        return header
    
    def _render_profile(self) -> str:
        """Render profile/summary section."""
        profile = self.sections.get("profile")
        if not profile:
            return ""
        return rf"""
\cvsection{{Profile}}
{self.fmt(profile)}
"""
    
    def _render_education(self) -> str:
        """Render education section."""
        education = self.sections.get("education", [])
        if not education:
            return ""
        
        entries = []
        for edu in education:
            degree = self.fmt(edu.get("degree", ""))
            institution = self.fmt(edu.get("institution", ""))
            date_range = self._format_date_range(edu.get("date_range", {}))
            
            # Build subentry info
            gpa = edu.get("gpa", {})
            ranking = edu.get("ranking", "")
            subentry_parts = []
            if gpa:
                subentry_parts.append(f"GPA: {gpa.get('score', '')}/{gpa.get('scale', '')}")
            if ranking:
                subentry_parts.append(self.fmt(ranking))
            subentry_text = "; ".join(subentry_parts)
            
            entry = rf"\entry{{{degree}}}{{{date_range}}}"
            entry += "\n" + rf"\subentry{{{institution}}}{{{subentry_text}}}"
            
            # Add thesis and other items
            items = []
            if edu.get("thesis_title"):
                items.append(rf"\item Thesis: \textit{{{self.fmt(edu['thesis_title'])}}}")
            if edu.get("thesis_summary"):
                items.append(rf"\item {self.fmt(edu['thesis_summary'])}")
            
            if items:
                entry += "\n\\begin{itemize}\n" + "\n".join(items) + "\n\\end{itemize}"
            
            entries.append(entry)
        
        return rf"""
\cvsection{{Education}}

{chr(10).join(entries)}
"""
    
    def _render_research_experience(self) -> str:
        """Render research experience section."""
        positions = self.sections.get("research_positions", [])
        if not positions:
            return ""
        
        entries = self._render_positions(positions)
        return rf"""
\cvsection{{Research Experience}}
{entries}
"""
    
    def _render_positions(self, positions: List[dict]) -> str:
        """Render a list of academic positions."""
        entries = []
        for pos in positions:
            title = self.fmt(pos.get("title", ""))
            org = self.fmt(pos.get("organization", ""))
            dept = self.fmt(pos.get("department_or_unit", ""))
            date_range = self._format_date_range(pos.get("date_range", {}))
            
            entry = rf"\entry{{{title}}}{{{date_range}}}"
            
            subentry_text = dept if dept else org
            if dept and org:
                subentry_text = f"{dept}, {org}"
            entry += "\n" + rf"\subentry{{{subentry_text}}}{{}}"
            
            bullets = pos.get("bullets", [])
            if bullets:
                items = [rf"\item {self.fmt(b)}" for b in bullets]
                entry += "\n\\begin{itemize}\n" + "\n".join(items) + "\n\\end{itemize}"
            
            entries.append(entry)
        
        return "\n\n".join(entries)
    
    def _render_research_interests(self) -> str:
        """Render research interests section."""
        interests = self.sections.get("research_interests", [])
        if not interests:
            return ""
        
        items = [rf"\item {self.fmt(i)}" for i in interests]
        return rf"""
\cvsection{{Research Interests}}
\begin{{itemize}}
{chr(10).join(items)}
\end{{itemize}}
"""
    
    def _render_professional_experience(self) -> str:
        """Render professional experience section."""
        positions = self.sections.get("professional_positions", [])
        clinical = self.sections.get("clinical_positions", [])
        
        all_positions = positions + clinical
        if not all_positions:
            return ""
        
        entries = self._render_positions(all_positions)
        return rf"""
\cvsection{{Professional Experience}}

{entries}
"""
    
    def _render_coursework(self) -> str:
        """Render selected coursework section."""
        coursework = self.sections.get("coursework", [])
        if not coursework:
            return ""
        
        items = []
        for course in coursework:
            title = self.fmt(course.get("title", ""))
            grade = course.get("grade_string", "")
            if grade:
                items.append(rf"\item {title} ({grade})")
            else:
                items.append(rf"\item {title}")
        
        return rf"""
\cvsection{{Selected Coursework (relevant \& high marks)}}
\begin{{itemize}}
{chr(10).join(items)}
\end{{itemize}}
"""
    
    def _render_publications(self) -> str:
        """Render publications section using BibTeX references."""
        publications = self.sections.get("publications", [])
        if not publications:
            return ""
        
        items = []
        for pub in publications:
            title = pub.get("title", "")
            status = pub.get("status", "")
            
            # Check if we have a cite key
            if title in self.cite_keys:
                cite_key = self.cite_keys[title]
                item = rf"\item \fullcitation{{{cite_key}}}"
                if status:
                    item += rf" \hfill[{self.fmt(status)}]"
                items.append(item)
            else:
                # Fallback: inline citation
                authors = ", ".join(pub.get("authors", []))
                venue = pub.get("venue", "")
                year = pub.get("year", "")
                citation = f"{self.fmt(authors)}. \\textit{{{self.fmt(title)}}}."
                if venue:
                    citation += f" {self.fmt(venue)}."
                if year:
                    citation += f" {year}."
                item = rf"\item {citation}"
                if status:
                    item += rf" \hfill[{self.fmt(status)}]"
                items.append(item)
        
        return rf"""
\cvsection{{Publications \& Presentations}}
\begin{{itemize}}
{chr(10).join(items)}
\end{{itemize}}
"""
    
    def _render_workshops(self) -> str:
        """Render workshops and certifications section."""
        workshops = self.sections.get("workshops_and_certifications", [])
        talks = self.sections.get("talks_and_presentations", [])
        
        all_items = []
        for ws in workshops:
            title = self.fmt(ws.get("title", ""))
            provider = ws.get("provider", "")
            if provider:
                all_items.append(rf"\item {title} ({self.fmt(provider)})")
            else:
                all_items.append(rf"\item {title}")
        
        if not all_items:
            return ""
        
        return rf"""
\cvsection{{Workshops \& Certifications}}
\begin{{itemize}}
{chr(10).join(all_items)}
\end{{itemize}}
"""
    
    def _render_skills(self) -> str:
        """Render skills section."""
        skills = self.sections.get("skills", [])
        if not skills:
            return ""
        
        items = []
        for skill in skills:
            topic = self.fmt(skill.get("topic", ""))
            skill_list = skill.get("skills", [])
            skill_str = ", ".join(self.fmt(s) for s in skill_list)
            items.append(rf"\item {topic}: {skill_str}.")
        
        return rf"""
\cvsection{{Skills}}
\begin{{itemize}}
{chr(10).join(items)}
\end{{itemize}}
"""
    
    def _render_memberships(self) -> str:
        """Render professional memberships section."""
        memberships = self.sections.get("memberships", [])
        if not memberships:
            return ""
        
        items = []
        for mem in memberships:
            org = self.fmt(mem.get("organization", ""))
            role = mem.get("role", "")
            if role:
                items.append(rf"\item {role}, {org}")
            else:
                items.append(rf"\item Member, {org}")
        
        return rf"""
\cvsection{{Professional Memberships}}
\begin{{itemize}}
{chr(10).join(items)}
\end{{itemize}}
"""
    
    def _render_languages(self) -> str:
        """Render language competence section."""
        languages = self.sections.get("languages", [])
        tests = self.sections.get("language_tests", [])
        
        if not languages and not tests:
            return ""
        
        # Render proficiency
        parts = []
        for lang in languages:
            name = self.fmt(lang.get("language", ""))
            level = lang.get("level", "")
            parts.append(rf"\textbf{{{name}:}} {level}")
        
        lang_line = " \\quad ".join(parts)
        
        # Render test scores table if present
        test_table = ""
        if tests:
            for test in tests:
                test_name = self.fmt(test.get("test_name", ""))
                overall = test.get("overall", "")
                section_scores = test.get("section_scores", {})
                
                if section_scores:
                    # Build table
                    headers = ["Overall"] + list(section_scores.keys())
                    scores = [str(overall or "")] + [str(v) for v in section_scores.values()]
                    
                    cols = " c" * len(headers)
                    header_row = " & ".join(rf"\textbf{{{h}}}" for h in headers)
                    score_row = " & ".join(scores)
                    
                    test_table += rf"""
\medskip
\begin{{tabular}}{{@{{}}l{cols}@{{}}}}
\toprule
\textbf{{{test_name}}} & {header_row}\\
\midrule
Score & {score_row}\\
\bottomrule
\end{{tabular}}
"""
        
        return rf"""
\cvsection{{Language Competence and Test Scores}}
{lang_line}
{test_table}
"""
    
    def _render_references(self) -> str:
        """Render references section."""
        references = self.sections.get("references", [])
        if not references:
            return ""
        
        rows = []
        for ref in references:
            name = self.fmt(ref.get("name", ""))
            email = ref.get("email", "")
            position = self.fmt(ref.get("position", ""))
            org = self.fmt(ref.get("organization", ""))
            
            email_link = rf"\textit{{\href{{mailto:{email}}}{{{email}}}}}" if email else ""
            pos_org = f"{position}, {org}" if position and org else (position or org)
            
            rows.append(rf"{name} & {email_link} & {pos_org}\\")
        
        rows_str = "\n".join(rows)
        
        return rf"""
\cvsection{{References}}
\renewcommand{{\arraystretch}}{{1.3}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}l l X@{{}}}}
\toprule
\textbf{{Name}} & \textbf{{Email}} & \textbf{{Position \& Organization}}\\
\midrule
{rows_str}
\bottomrule
\end{{tabularx}}
"""

    @staticmethod
    def _render_closing() -> str:
        """Render document closing."""
        return r"""
\end{document}
"""
