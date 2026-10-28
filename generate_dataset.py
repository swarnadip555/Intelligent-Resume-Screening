"""
Dataset Generator for Intelligent Resume Screening & Job Matching System.
Generates 250 realistic, synthetic candidate profiles across 4 job categories:
- Data Analyst
- Software Developer
- Data Scientist
- IT Support

Follows Responsible AI guidelines:
- Strictly fictional data
- No personal identifying information (PII) such as real names, phone numbers, emails, addresses, or photos
- Balanced representation across categories
"""

import os
import random
import pandas as pd
import numpy as np

# Set fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

# Taxonomy of skills per domain
CATEGORY_SKILLS = {
    "Data Analyst": [
        "Python", "SQL", "Pandas", "NumPy", "Tableau", "Power BI", "Excel",
        "Data Visualization", "ETL", "Statistical Analysis", "Business Intelligence",
        "A/B Testing", "Reporting", "Matplotlib", "Seaborn", "PostgreSQL",
        "Google Analytics", "Data Cleaning", "Dashboard Development"
    ],
    "Software Developer": [
        "Python", "JavaScript", "TypeScript", "React", "Node.js", "Java",
        "C++", "C#", "Git", "Docker", "REST APIs", "GraphQL", "Microservices",
        "PostgreSQL", "MongoDB", "Redis", "AWS", "CI/CD", "Unit Testing",
        "Agile", "Linux", "Kubernetes", "HTML/CSS"
    ],
    "Data Scientist": [
        "Python", "R", "SQL", "Machine Learning", "Deep Learning", "TensorFlow",
        "PyTorch", "Scikit-Learn", "NLP", "Pandas", "NumPy", "Statistics",
        "Feature Engineering", "Data Modeling", "Big Data", "Spark",
        "Computer Vision", "Transformers", "Model Deployment", "MLOps"
    ],
    "IT Support": [
        "Active Directory", "Windows Server", "Linux Administration", "Troubleshooting",
        "Hardware Diagnostics", "Networking", "TCP/IP", "DNS", "DHCP", "VPN",
        "Help Desk", "ServiceNow", "Office 365", "System Administration",
        "Backup & Recovery", "Security Patching", "ITIL", "Customer Service"
    ]
}

EDUCATION_LEVELS = [
    "Bachelor's in Computer Science",
    "Bachelor's in Information Technology",
    "Bachelor's in Data Science",
    "Bachelor's in Mathematics",
    "Master's in Data Science",
    "Master's in Computer Science",
    "Master's in Business Analytics",
    "Bachelor's in Business Administration",
    "Associate Degree in Network Administration",
    "PhD in Computer Science"
]

CATEGORY_EXPERIENCE_RANGES = {
    "Data Analyst": (1, 10),
    "Software Developer": (1, 14),
    "Data Scientist": (1, 12),
    "IT Support": (1, 12)
}

PROJECT_TEMPLATES = {
    "Data Analyst": [
        "Developed executive KPI dashboards in {viz_tool} tracking customer retention, increasing stakeholder visibility by 35%.",
        "Automated recurring reporting pipelines using Python and {db_tool}, reducing manual ETL effort by 15 hours weekly.",
        "Conducted exploratory data analysis on 2M+ transaction records using Pandas and SQL to identify key revenue drivers.",
        "Formulated statistical A/B test methodologies to evaluate UI changes, driving a 12% boost in customer conversion rates.",
        "Built predictive customer segmentation models to uncover high-value user clusters for targeted marketing campaigns."
    ],
    "Software Developer": [
        "Engineered scalable backend RESTful microservices in {lang} handling 50k+ daily active requests with 99.9% uptime.",
        "Built responsive single-page frontend application utilizing {frontend_tool} and state management libraries.",
        "Containerized legacy monolithic services with Docker and set up automated CI/CD deployment pipelines on {cloud_tool}.",
        "Optimized database indexing and caching strategies with Redis and {db_tool}, slashing API response latency by 45%.",
        "Collaborated in Agile sprints following test-driven development (TDD), maintaining unit test coverage exceeding 85%."
    ],
    "Data Scientist": [
        "Designed and deployed end-to-end Machine Learning pipelines using Scikit-Learn and {ml_framework} for predictive risk modeling.",
        "Trained Natural Language Processing models with Transformers for multi-class customer review sentiment classification.",
        "Engineered robust feature transformation pipelines and handled high-cardinality categorical features on distributed datasets.",
        "Constructed deep learning recommendation architectures improving CTR by 18% in production A/B tests.",
        "Spearheaded MLOps monitoring infrastructure to detect data drift and automate model retraining cycles."
    ],
    "IT Support": [
        "Provided Tier 2/3 technical support across Windows and {os_tool} environments, resolving 95%+ of SLA incident tickets in ServiceNow.",
        "Configured user permissions, group policies, and identity access management via Active Directory and Microsoft 365.",
        "Diagnosed complex LAN/WAN network connectivity faults, configuring routers, switches, DHCP, and secure remote VPN tunnels.",
        "Conducted automated hardware audits, system imaging, and scheduled security vulnerability patch management cycles.",
        "Authored internal IT knowledge base documentation that accelerated end-user self-service resolution by 30%."
    ]
}

SUMMARY_TEMPLATES = {
    "Data Analyst": [
        "Detail-oriented Data Analyst with {exp} years of experience transforming raw datasets into actionable commercial intelligence.",
        "Analytical professional with {exp} years specializing in SQL querying, statistical modeling, and interactive business reporting.",
        "Data-driven specialist with {exp} years of proven expertise in ETL automation, predictive analytics, and executive dashboard design."
    ],
    "Software Developer": [
        "Software Engineer with {exp} years of hands-on experience architecting full-stack web applications and robust distributed systems.",
        "Results-oriented Software Developer with {exp} years building scalable REST APIs, microservices, and modern frontend interfaces.",
        "Full-Stack Developer with {exp} years experienced in cloud infrastructure, automated CI/CD pipelines, and high-performance backend code."
    ],
    "Data Scientist": [
        "Data Scientist with {exp} years of track record applying machine learning, statistical theory, and deep learning algorithms to business challenges.",
        "Quantitative researcher and ML Practitioner with {exp} years designing predictive models, NLP systems, and scalable MLOps architectures.",
        "Machine Learning Engineer with {exp} years of expertise spanning exploratory modeling, transformer fine-tuning, and production deployment."
    ],
    "IT Support": [
        "IT Support Specialist with {exp} years providing tier-1 through tier-3 technical assistance, enterprise system administration, and network troubleshooting.",
        "Proactive Systems Support Administrator with {exp} years of experience maintaining Active Directory, cloud services, and endpoint security.",
        "Customer-focused IT Technician with {exp} years resolving enterprise hardware, software, VPN, and infrastructure incidents."
    ]
}

def generate_resume(candidate_id, category):
    min_exp, max_exp = CATEGORY_EXPERIENCE_RANGES[category]
    years_exp = int(np.clip(np.random.normal(loc=(min_exp + max_exp)/2, scale=2.5), min_exp, max_exp))
    
    # Pick 5-9 skills relevant to the category, occasionally with 1 cross-domain skill
    cat_skills = list(CATEGORY_SKILLS[category])
    num_skills = random.randint(5, 9)
    selected_skills = random.sample(cat_skills, k=min(num_skills, len(cat_skills)))
    
    # Add a slight cross-pollination (e.g., Data Scientist knowing Docker or SQL)
    if random.random() < 0.35:
        cross_category = random.choice([c for c in CATEGORY_SKILLS if c != category])
        cross_skill = random.choice(CATEGORY_SKILLS[cross_category])
        if cross_skill not in selected_skills:
            selected_skills.append(cross_skill)
            
    random.shuffle(selected_skills)
    skills_str = ";".join(selected_skills)
    
    # Education
    education = random.choice(EDUCATION_LEVELS)
    
    # Build resume text
    summary = random.choice(SUMMARY_TEMPLATES[category]).format(exp=years_exp)
    
    # Projects/responsibilities
    proj_templates = random.sample(PROJECT_TEMPLATES[category], k=3)
    projects = []
    for pt in proj_templates:
        proj = pt.format(
            viz_tool=random.choice(["Tableau", "Power BI", "Seaborn", "Looker"]),
            db_tool=random.choice(["PostgreSQL", "MySQL", "Snowflake", "SQL Server"]),
            lang=random.choice(["Python", "Java", "Go", "TypeScript", "C#"]),
            frontend_tool=random.choice(["React", "TypeScript", "Next.js", "Vue"]),
            cloud_tool=random.choice(["AWS", "Azure", "GCP"]),
            ml_framework=random.choice(["PyTorch", "TensorFlow", "XGBoost"]),
            os_tool=random.choice(["Linux", "Windows Server", "macOS"])
        )
        projects.append(proj)
        
    resume_text = (
        f"Summary: {summary} "
        f"Core Competencies: {', '.join(selected_skills)}. "
        f"Professional Experience: {' '.join(projects)} "
        f"Education: {education}."
    )
    
    return {
        "candidate_id": candidate_id,
        "resume_text": resume_text,
        "years_experience": years_exp,
        "skills": skills_str,
        "education_level": education.split(" in ")[0],  # Bachelor's, Master's, PhD, etc.
        "job_category": category
    }

def main():
    os.makedirs("data", exist_ok=True)
    records = []
    categories = ["Data Analyst", "Software Developer", "Data Scientist", "IT Support"]
    num_records_per_cat = 62  # 62 * 4 = 248 records (well within 100-500)
    
    cid = 1
    for cat in categories:
        for _ in range(num_records_per_cat):
            candidate_id = f"C{cid:03d}"
            records.append(generate_resume(candidate_id, cat))
            cid += 1
            
    # Add a couple of border cases / additional records to reach 250
    records.append(generate_resume(f"C{cid:03d}", "Data Analyst"))
    cid += 1
    records.append(generate_resume(f"C{cid:03d}", "Data Scientist"))
    
    random.shuffle(records)
    
    df = pd.DataFrame(records)
    output_path = os.path.join("data", "resumes.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} synthetic candidate profiles at: {output_path}")
    print("\nJob Category Breakdown:")
    print(df["job_category"].value_counts())
    print("\nYears Experience Summary:")
    print(df["years_experience"].describe())

if __name__ == "__main__":
    main()
