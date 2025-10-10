import csv
import heapq
from difflib import SequenceMatcher

# ========== Heuristic Function ==========
def text_similarity(a, b):
    """Compute a basic similarity ratio between two strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def heuristic(query, medicine):
    """Compute combined similarity score between query and medicine fields."""
    name_sim = text_similarity(query.get("name", ""), medicine.get("Name", ""))
    cat_sim = text_similarity(query.get("category", ""), medicine.get("Category", ""))
    ind_sim = text_similarity(query.get("indication", ""), medicine.get("Indication", ""))
    return (name_sim + cat_sim + ind_sim) / 3.0  # weighted average


# ========== Best-First Search ==========
def best_first_search(query, medicines, top_k=5):
    """Return top-k medicines based on similarity."""
    pq = []  # priority queue (max-heap by negative score)
    for med in medicines:
        score = heuristic(query, med)
        heapq.heappush(pq, (-score, med))

    top_matches = []
    for _ in range(min(top_k, len(pq))):
        score, med = heapq.heappop(pq)
        med["Similarity Score"] = round(-score, 3)
        top_matches.append(med)
    return top_matches


# ========== Data Loader ==========
def load_medicines(csv_file):
    """Load medicine dataset from CSV file."""
    with open(csv_file, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        medicines = []
        for row in reader:
            # Strip whitespace from keys and values to avoid mismatch
            cleaned = {k.strip(): v.strip() for k, v in row.items() if k}
            medicines.append(cleaned)
        return medicines


# ========== Main Program ==========
if __name__ == "__main__":
    medicines = load_medicines("medicine_dataset.csv")

    print("\n=== 🩺 Medicine Lookup System ===")
    name = input("Enter medicine name (or leave blank): ").strip()
    category = input("Enter category (or leave blank): ").strip()
    indication = input("Enter indication (or leave blank): ").strip()

    query = {"name": name, "category": category, "indication": indication}
    results = best_first_search(query, medicines, top_k=5)

    print("\n🔎 Top Matches:\n")
    if results:
        print(f"{'Name':<15}{'Category':<15}{'Dosage Form':<15}{'Strength':<10}"
              f"{'Manufacturer':<25}{'Indication':<15}{'Classification':<15}")
        print("-" * 110)
        for med in results:
            print(f"{med.get('Name',''):<15}{med.get('Category',''):<15}{med.get('Dosage Form',''):<15}"
                  f"{med.get('Strength',''):<10}{med.get('Manufacturer',''):<25}{med.get('Indication',''):<15}"
                  f"{med.get('Classification',''):<15}")
    else:
        print("No matching medicines found.")
