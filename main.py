# Drugs, Side Effects and Medical Conditions Analysis
# Complete Data Analysis Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Configure visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def main():
    print("="*60)
    print("DRUGS, SIDE EFFECTS AND MEDICAL CONDITIONS ANALYSIS")
    print("="*60)
    
    try:
        # Load the dataset
        print("\n1. 📂 LOADING DATASET...")
        df = pd.read_csv('data/drugs_side_effects_drugs_com.csv')
        print(f"   ✅ Dataset loaded successfully!")
        print(f"   📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Data cleaning
        print("\n2. 🧹 CLEANING DATA...")
        df_clean = clean_data(df)
        print("   ✅ Data cleaning completed!")
        
        # Exploratory Data Analysis
        print("\n3. 🔍 PERFORMING ANALYSIS...")
        perform_analysis(df_clean)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE! 🎉")
        print("="*60)
        print(f"Total drugs analyzed: {len(df_clean)}")
        print(f"Medical conditions: {df_clean['medical_condition'].nunique()}")
        print(f"Unique drug classes: {df_clean['drug_classes'].nunique()}")
        
    except FileNotFoundError:
        print("❌ ERROR: Dataset file not found!")
        print("Please ensure 'drugs_side_effects_drugs_com.csv' is in the 'data/' folder")
        print("Download from: https://www.kaggle.com/datasets/amaanansari09/drugs-side-effects-and-medical-condition")

def clean_data(df):
    """Clean and preprocess the dataset"""
    df_clean = df.copy()
    
    # Handle missing values
    df_clean['side_effects'] = df_clean['side_effects'].fillna('Unknown')
    df_clean['related_drugs'] = df_clean['related_drugs'].fillna('Unknown')
    df_clean['generic_name'] = df_clean['generic_name'].fillna('Unknown')
    df_clean['drug_classes'] = df_clean['drug_classes'].fillna('Unknown')
    df_clean['pregnancy_category'] = df_clean['pregnancy_category'].fillna('Unknown')
    df_clean['rx_otc'] = df_clean['rx_otc'].fillna('Unknown')
    
    # Numerical columns
    df_clean['rating'] = df_clean['rating'].fillna(0)
    df_clean['no_of_reviews'] = df_clean['no_of_reviews'].fillna(0)
    
    # Convert activity to numerical
    df_clean['activity'] = df_clean['activity'].str.rstrip('%').astype('float') / 100
    
    # Convert alcohol interaction to binary
    df_clean['alcohol'] = df_clean['alcohol'].apply(lambda x: 1 if x == 'X' else 0)
    
    return df_clean

def perform_analysis(df_clean):
    """Perform comprehensive data analysis"""
    
    # 1. Medical Conditions Analysis
    print("\n   🏥 Analyzing medical conditions...")
    condition_counts = df_clean['medical_condition'].value_counts().head(10)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x=condition_counts.values, y=condition_counts.index, palette="viridis")
    plt.title('Top 10 Most Common Medical Conditions', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Drugs')
    plt.ylabel('Medical Condition')
    plt.tight_layout()
    plt.savefig('results/top_medical_conditions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Side Effects Analysis
    print("   ⚠️  Analyzing side effects...")
    all_effects = []
    for effects_list in df_clean['side_effects'].apply(extract_common_effects):
        all_effects.extend(effects_list)
    
    effect_counts = Counter(all_effects).most_common(10)
    effects_df = pd.DataFrame(effect_counts, columns=['Side_Effect', 'Count'])
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Count', y='Side_Effect', data=effects_df, palette="rocket")
    plt.title('Top 10 Most Common Side Effects', fontsize=16, fontweight='bold')
    plt.xlabel('Frequency')
    plt.ylabel('Side Effect')
    plt.tight_layout()
    plt.savefig('results/top_side_effects.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Rx vs OTC Analysis
    print("   💊 Analyzing prescription patterns...")
    rx_otc_counts = df_clean['rx_otc'].value_counts()
    
    plt.figure(figsize=(10, 8))
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    plt.pie(rx_otc_counts.values, labels=rx_otc_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Distribution of Rx vs OTC Drugs', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.savefig('results/rx_otc_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Pregnancy Category Analysis
    print("   🤰 Analyzing pregnancy safety...")
    pregnancy_counts = df_clean['pregnancy_category'].value_counts()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=pregnancy_counts.values, y=pregnancy_counts.index, palette="coolwarm")
    plt.title('Drugs by Pregnancy Safety Category', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Drugs')
    plt.ylabel('Pregnancy Category')
    plt.tight_layout()
    plt.savefig('results/pregnancy_categories.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Rating Analysis
    print("   ⭐ Analyzing drug ratings...")
    valid_ratings = df_clean[df_clean['rating'] > 0]
    
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(valid_ratings['rating'], bins=20, kde=True)
    plt.title('Distribution of Drug Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='pregnancy_category', y='rating', data=valid_ratings)
    plt.title('Ratings by Pregnancy Category')
    plt.xlabel('Pregnancy Category')
    plt.ylabel('Rating')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/rating_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Save results to CSV
    print("   💾 Saving analysis results...")
    condition_counts.to_csv('results/medical_condition_counts.csv')
    effects_df.to_csv('results/side_effect_counts.csv')
    rx_otc_counts.to_csv('results/rx_otc_counts.csv')
    
    print("   ✅ All analyses completed and results saved!")

def extract_common_effects(text):
    """Extract common side effects from text"""
    if text == 'Unknown':
        return []
    
    effects = []
    text_lower = text.lower()
    
    common_effects = ['hives', 'itching', 'nausea', 'dizziness', 'headache', 
                     'drowsiness', 'vomiting', 'rash', 'diarrhea', 'constipation',
                     'breathing', 'swelling', 'fever', 'pain', 'fatigue']
    
    for effect in common_effects:
        if effect in text_lower:
            effects.append(effect)
    
    return effects

if __name__ == "__main__":
    main()