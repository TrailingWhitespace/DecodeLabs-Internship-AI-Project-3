#  DecodeLabs AI Project 2: Tech stack recommender

#  Architecture: IPO Model
#    INPUT  -> User enters 3 or more skills to get recommendations
#    PROCESS -> Use TF-IDF to convert skills into weighted vectors,
#               then calculate cosine similarity between user vector
#               and all job role vectors in the dataset
#    OUTPUT -> Top 3 most relevant job roles ranked by match score

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("raw_skills.csv")


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["skills"])

print("\nEnter your skills (minimum 3) to get career recommendations.")
print("Use underscores for multi-word skills (e.g. machine_learning, deep_learning)")
print("-" * 50)

user_skills = []

while len(user_skills) < 3:
    remaining = 3 - len(user_skills)
    skill = input(f"Enter skill {len(user_skills) + 1}: ").strip().lower().replace(" ", "_")
    if skill:
        user_skills.append(skill)
    if len(user_skills) < 3:
        print(f"  (Need at least {remaining} more skill(s))")


print("\nYou can add more skills for better accuracy (press Enter to skip):")
while True:
    skill = input(f"Enter skill {len(user_skills) + 1} (or press Enter to finish): ").strip().lower().replace(" ", "_")
    if skill == "":
        break
    user_skills.append(skill)

print(f"\nYour skills: {', '.join(user_skills)}")


user_input_string = " ".join(user_skills)


user_vector = vectorizer.transform([user_input_string])


similarity_scores = cosine_similarity(user_vector, tfidf_matrix)



df["match_score"] = similarity_scores[0]
# print(similarity_scores)
df_sorted = df.sort_values("match_score", ascending=False)


top_n = 3
top_results = df_sorted.head(top_n)

print("\n" + "=" * 50)
print(f"   TOP {top_n} CAREER RECOMMENDATIONS FOR YOU")
print("=" * 50)

for rank, (_, row) in enumerate(top_results.iterrows(), start=1):
    score_percent = round(row["match_score"] * 100, 2)
    print(f"\n#{rank}  {row['job_role']}")
    print(f"    Match Score : {score_percent}%")
    print(f"    Key Skills  : {row['skills'].replace('_', ' ')}")

