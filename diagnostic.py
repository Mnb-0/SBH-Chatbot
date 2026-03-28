import pandas as pd

def analyze_audit_results(csv_path="audited_corporate_test.csv"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return

    total_queries = len(df)
    
    # Check if the Hallucinated column exists (handling potential boolean parsing)
    if 'Hallucinated' not in df.columns:
        print("Error: 'Hallucinated' column missing. Check your CSV structure.")
        return
        
    # Convert to boolean just in case pandas read them as strings
    df['Hallucinated'] = df['Hallucinated'].astype(bool)
    
    total_hallucinations = df['Hallucinated'].sum()
    failure_rate = (total_hallucinations / total_queries) * 100

    print("📊 --- RAG AUDIT REPORT --- 📊")
    print(f"Total Queries Tested: {total_queries}")
    print(f"Total Hallucinations Flagged: {total_hallucinations}")
    print(f"Pipeline Failure Rate: {failure_rate:.1f}%\n")

    print("🏢 --- FAILURES BY SECTOR --- 🏢")
    sector_fails = df.groupby('Routed_Sector')['Hallucinated'].mean() * 100
    for sector, rate in sector_fails.items():
        print(f"{sector:15} : {rate:.1f}% failure rate")

    fails = df[df['Hallucinated']]
    if not fails.empty:
        print("\n🚨 --- THE CASUALTIES (Failed Audits) --- 🚨")
        for _, row in fails.iterrows():
            print(f"Sector : {row['Routed_Sector']}")
            print(f"Query  : {row['Question']}")
            print(f"Reason : {row['Audit_Reason']}")
            print("-" * 50)
    else:
        print("\n✅ Zero hallucinations detected. The pipeline is rock solid.")

if __name__ == "__main__":
    analyze_audit_results()