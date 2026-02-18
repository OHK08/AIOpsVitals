import csv
import heapq
from difflib import SequenceMatcher


# ========== Heuristic Function ==========
def text_similarity(a, b):
    """Compute similarity between two strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def heuristic(query, medicine):
    """Compute similarity score using Chemical_Name, Category, Indication."""
    chem_sim = text_similarity(query.get("chemical", ""), medicine.get("Chemical_Name", ""))
    cat_sim = text_similarity(query.get("category", ""), medicine.get("Category", ""))
    ind_sim = text_similarity(query.get("indication", ""), medicine.get("Indication", ""))

    # Weighted average (chemical name is most important)
    return (0.5 * chem_sim + 0.3 * cat_sim + 0.2 * ind_sim)


# ========== Best-First Search ==========
def best_first_search(query, medicines, top_k=5):
    pq = []
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
    with open(csv_file, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        medicines = []
        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items() if k}
            medicines.append(cleaned)
        return medicines


# ========== Main Program ==========
if __name__ == "__main__":
    medicines = load_medicines("medicines_with_chemical.csv")

    print("\n=== 🧪 Chemical-Based Medicine Lookup System ===")

    chemical = input("Enter chemical name (e.g., Paracetamol, Ibuprofen): ").strip()
    category = input("Enter category (or leave blank): ").strip()
    indication = input("Enter indication (or leave blank): ").strip()

    query = {"chemical": chemical, "category": category, "indication": indication}
    results = best_first_search(query, medicines, top_k=5)

    print("\n🔎 Top Matches:\n")
    if results:
        print(f"{'Brand Name':<15}{'Chemical':<15}{'Category':<15}{'Dosage Form':<15}"
              f"{'Strength':<10}{'Manufacturer':<25}{'Indication':<15}{'Class':<15}{'Score':<6}")
        print("-" * 130)

        for med in results:
            print(f"{med.get('Name', ''):<15}{med.get('Chemical_Name', ''):<15}{med.get('Category', ''):<15}"
                  f"{med.get('Dosage Form', ''):<15}{med.get('Strength', ''):<10}"
                  f"{med.get('Manufacturer', ''):<25}{med.get('Indication', ''):<15}"
                  f"{med.get('Classification', ''):<15}{med.get('Similarity Score', ''):<6}")
    else:
        print("No matching medicines found.")
