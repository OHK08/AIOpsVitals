import pandas as pd

# Load your dataset
df = pd.read_csv("medicine_dataset.csv")

chemical_map = {
    "Acetostatin": "Paracetamol",
    "Acetomycin": "Amoxicillin",
    "Dolocillin": "Ibuprofen",
    "Ibuprocillin": "Ibuprofen",
    "Dextrophen": "Dextromethorphan",
    "Dolophen": "Paracetamol",
    "Dextromet": "Dextromethorphan",
    "Doloprofen": "Ibuprofen",
    "Metonazole": "Metronidazole",
    "Dolomycin": "Amoxicillin",
    "Metostatin": "Metformin",
    "Clarinazole": "Clarithromycin",
    "Acetonazole": "Ketoconazole",
    "Cefvir": "Cefixime",
    "Ibupromet": "Ibuprofen",
    "Amoxivir": "Amoxicillin",
    "Metovir": "Metformin",
    "Amoxinazole": "Amoxicillin",
    "Metoprofen": "Metformin",
    "Dextrovir": "Dextromethorphan",
    "Clarimet": "Clarithromycin",
    "Dextrocillin": "Amoxicillin",
    "Metomycin": "Metformin",
    "Cefcillin": "Cefixime",
    "Cefmycin": "Ceftriaxone",
    "Metocillin": "Metformin",
    "Dextronazole": "Metronidazole",
    "Dextromycin": "Azithromycin",
    "Clarivir": "Clarithromycin",
    "Cefphen": "Cefixime",
    "Metomet": "Metformin",
    "Amoxistatin": "Amoxicillin",
    "Dextrostatin": "Dextromethorphan",
    "Dextroprofen": "Ibuprofen",
    "Ibuproprofen": "Ibuprofen",
    "Cefprofen": "Cefixime",
    "Acetovir": "Paracetamol",
    "Amoxiprofen": "Amoxicillin",
    "Cefmet": "Cefixime",
    "Acetomet": "Paracetamol",
    "Clariprofen": "Clarithromycin",
    "Ibuprovir": "Ibuprofen",
    "Acetophen": "Paracetamol",
    "Dolovir": "Paracetamol",
    "Metophen": "Metformin",
    "Amoximet": "Amoxicillin",
    "Ibupromycin": "Azithromycin",
    "Clariphen": "Clarithromycin",
    "Dolostatin": "Ibuprofen",
    "Dolonazole": "Metronidazole",
    "Cefstatin": "Cefixime",
    "Clarimycin": "Clarithromycin",
    "Amoximycin": "Amoxicillin",
    "Ibuprophen": "Ibuprofen",
    "Dolomet": "Paracetamol",
    "Cefnazole": "Cefixime",
    "Acetocillin": "Paracetamol",
    "Acetoprofen": "Ibuprofen",
    "Claricillin": "Clarithromycin",
    "Amoxiphen": "Amoxicillin",
    "Ibupronazole": "Ibuprofen",
    "Ibuprostatin": "Ibuprofen",
    "Claristatin": "Clarithromycin",
    "Amoxicillin": "Amoxicillin"
}


# Add Chemical_Name column
df["Chemical_Name"] = df["Name"].map(chemical_map)

# Fill missing values if any
df["Chemical_Name"].fillna("Unknown", inplace=True)

# Save updated dataset
df.to_csv("medicines_with_chemical.csv", index=False)

print("Chemical Name column added successfully!")
