"""
============================================================
Synthetic Student Dataset Generator
============================================================
Project : Predict Success — Graduation & Placement Forecasting
Purpose : Generates a realistic synthetic student dataset and
          saves it as a CSV file for use in the main notebook.
============================================================
"""

import numpy as np
import pandas as pd
import os

# ── Configuration ──────────────────────────────────────────
N_STUDENTS = 1000
RANDOM_SEED = 42
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "student_data.csv")

# ── Set seed for reproducibility ───────────────────────────
np.random.seed(RANDOM_SEED)


def generate_dataset(n: int = N_STUDENTS) -> pd.DataFrame:
    """Generate a synthetic student dataset with realistic correlations."""

    # --- Unique student identifiers ---
    student_ids = [f"STU{str(i).zfill(4)}" for i in range(1, n + 1)]

    # --- Demographic features ---
    genders = np.random.choice(["Male", "Female"], size=n, p=[0.55, 0.45])
    ages = np.random.randint(18, 27, size=n)

    # --- Academic features ---
    attendance = np.clip(
        np.random.normal(75, 15, size=n), 20, 100
    ).round(1)

    cgpa = np.clip(
        np.random.normal(7.0, 1.5, size=n), 2.0, 10.0
    ).round(2)

    internal_marks = np.clip(
        np.random.normal(55, 15, size=n), 10, 100
    ).round(1)

    external_marks = np.clip(
        np.random.normal(50, 18, size=n), 5, 100
    ).round(1)

    backlogs = np.random.choice(
        [0, 0, 0, 0, 1, 1, 2, 3, 4, 5], size=n
    )

    study_hours = np.clip(
        np.random.normal(15, 6, size=n), 1, 40
    ).round(1)

    # --- Skill / Experience features ---
    internship_exp = np.random.choice(
        [0, 1], size=n, p=[0.45, 0.55]
    )

    projects = np.random.choice(
        [0, 1, 2, 3, 4, 5], size=n,
        p=[0.10, 0.20, 0.30, 0.20, 0.15, 0.05]
    )

    comm_skills = np.clip(
        np.random.normal(6.5, 2.0, size=n), 1, 10
    ).round(1)

    aptitude_score = np.clip(
        np.random.normal(60, 18, size=n), 10, 100
    ).round(1)

    # ── Target 1 — Graduation Status ──────────────────────
    # Depends heavily on academic metrics (CGPA, attendance,
    # marks, backlogs, study hours).
    grad_score = (
        0.30 * (cgpa / 10)
        + 0.25 * (attendance / 100)
        + 0.15 * (internal_marks / 100)
        + 0.10 * (study_hours / 40)
        - 0.20 * (backlogs / 5)
        + np.random.normal(0, 0.08, size=n)
    )
    graduation_status = (
        grad_score > np.percentile(grad_score, 25)
    ).astype(int)

    # ── Target 2 — Placement Status ───────────────────────
    # Depends on a mix of academics + skills + experience.
    place_score = (
        0.25 * (cgpa / 10)
        + 0.15 * (aptitude_score / 100)
        + 0.15 * (comm_skills / 10)
        + 0.15 * internship_exp
        + 0.10 * (projects / 5)
        + 0.10 * (attendance / 100)
        - 0.10 * (backlogs / 5)
        + np.random.normal(0, 0.10, size=n)
    )
    placement_status = (
        place_score > np.percentile(place_score, 35)
    ).astype(int)

    # ── Assemble DataFrame ────────────────────────────────
    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Gender": genders,
        "Age": ages,
        "Attendance_Percentage": attendance,
        "CGPA": cgpa,
        "Internal_Marks": internal_marks,
        "External_Marks": external_marks,
        "Backlogs": backlogs,
        "Study_Hours_Per_Week": study_hours,
        "Internship_Experience": internship_exp,
        "Projects_Completed": projects,
        "Communication_Skills_Score": comm_skills,
        "Aptitude_Test_Score": aptitude_score,
        "Graduation_Status": graduation_status,
        "Placement_Status": placement_status,
    })

    # ── Introduce realistic imperfections ─────────────────
    # ~2 % missing values in selected columns
    for col in [
        "Attendance_Percentage",
        "Internal_Marks",
        "Communication_Skills_Score",
        "Aptitude_Test_Score",
    ]:
        missing_idx = np.random.choice(
            df.index, size=int(n * 0.02), replace=False
        )
        df.loc[missing_idx, col] = np.nan

    # A handful of duplicate rows
    dup_rows = df.sample(n=5, random_state=RANDOM_SEED)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


# ── Main execution ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Synthetic Student Dataset Generator")
    print("=" * 60)

    df = generate_dataset(N_STUDENTS)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Dataset generated successfully!")
    print(f"   Records : {df.shape[0]}")
    print(f"   Features: {df.shape[1]}")
    print(f"   Saved to: {OUTPUT_FILE}")
    print(f"\n📊 Class distribution:")
    print(f"   Graduation — {dict(df['Graduation_Status'].value_counts())}")
    print(f"   Placement  — {dict(df['Placement_Status'].value_counts())}")
    print(f"   Missing    — {df.isnull().sum().sum()} cells")
    print(f"   Duplicates — 5 rows (intentional)")
    print("=" * 60)
