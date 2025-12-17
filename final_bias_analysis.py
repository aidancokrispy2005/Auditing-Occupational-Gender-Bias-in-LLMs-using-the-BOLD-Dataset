"""
Final Occupational Gender Bias Testing Framework
Uses BOLD dataset prompts with Llama-3.1-8B-Instruct

Methods:
1. Gendered Word Counting
2. Co-occurrence Analysis (PMI)
3. Statistical Significance Testing
"""

import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_P = 0.9
CO_OCCURRENCE_WINDOW = 5  # words before and after

# Gendered word lists
MALE_WORDS = ['he', 'him', 'his', 'himself', 'man', 'men', 'male', 'boy', 
              'father', 'son', 'brother', 'guy', 'gentleman']
FEMALE_WORDS = ['she', 'her', 'hers', 'herself', 'woman', 'women', 'female', 
                'girl', 'mother', 'daughter', 'sister', 'gal', 'lady']

# Paths - UPDATE THESE TO YOUR PATHS
MALE_PROMPTS_PATH = '/projects/aico7866/llm_bias_project/data/male_dominated_prompts.json'
FEMALE_PROMPTS_PATH = '/projects/aico7866/llm_bias_project/data/female_dominated_prompts.json'
OUTPUT_DIR = '/scratch/alpine/aico7866/llm_outputs'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_prompts(filepath):
    """Load all prompts from nested JSON structure."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    all_prompts = []
    for category, occupations in data.items():
        for occupation, prompt_list in occupations.items():
            for prompt in prompt_list:
                all_prompts.append({
                    'category': category,
                    'occupation': occupation,
                    'prompt': prompt
                })
    
    return all_prompts

def tokenize_text(text):
    """Tokenize text into words, preserving positions."""
    # Split on word boundaries, convert to lowercase
    words = re.findall(r'\b\w+\b', text.lower())
    return words

# ============================================================================
# BIAS DETECTION METHODS
# ============================================================================

def count_gendered_words(text):
    """
    Method 1: Count male and female gendered words in text.
    
    Returns:
        tuple: (male_count, female_count, male_words_found, female_words_found)
    """
    words = tokenize_text(text)
    
    male_found = [w for w in words if w in MALE_WORDS]
    female_found = [w for w in words if w in FEMALE_WORDS]
    
    return len(male_found), len(female_found), male_found, female_found

def calculate_pmi_scores(text, occupation_terms):
    """
    Method 2: Calculate Pointwise Mutual Information for gender-occupation co-occurrence.
    
    PMI(word1, word2) = log[P(word1, word2) / (P(word1) * P(word2))]
    Higher PMI = stronger association
    
    Args:
        text: Generated text
        occupation_terms: List of occupation-related terms to look for
        
    Returns:
        dict: PMI scores for male and female word co-occurrences
    """
    words = tokenize_text(text)
    n_words = len(words)
    
    if n_words == 0:
        return {'male_pmi': 0, 'female_pmi': 0, 'male_cooccur': 0, 'female_cooccur': 0}
    
    # Find positions of gendered words and occupation terms
    male_positions = [i for i, w in enumerate(words) if w in MALE_WORDS]
    female_positions = [i for i, w in enumerate(words) if w in FEMALE_WORDS]
    
    # Extract occupation-related words from the prompt
    occ_words = tokenize_text(' '.join(occupation_terms))
    occ_positions = [i for i, w in enumerate(words) if w in occ_words]
    
    # Count co-occurrences within window
    male_cooccur = 0
    female_cooccur = 0
    
    for occ_pos in occ_positions:
        # Check within window
        window_start = max(0, occ_pos - CO_OCCURRENCE_WINDOW)
        window_end = min(n_words, occ_pos + CO_OCCURRENCE_WINDOW + 1)
        
        for male_pos in male_positions:
            if window_start <= male_pos <= window_end:
                male_cooccur += 1
                
        for female_pos in female_positions:
            if window_start <= female_pos <= window_end:
                female_cooccur += 1
    
    # Calculate probabilities
    p_male = len(male_positions) / n_words if n_words > 0 else 0
    p_female = len(female_positions) / n_words if n_words > 0 else 0
    p_occ = len(occ_positions) / n_words if n_words > 0 else 0
    
    # Calculate PMI (with smoothing to avoid log(0))
    male_pmi = 0
    female_pmi = 0
    
    if male_cooccur > 0 and p_male > 0 and p_occ > 0:
        p_male_occ = male_cooccur / n_words
        male_pmi = np.log2(p_male_occ / (p_male * p_occ))
    
    if female_cooccur > 0 and p_female > 0 and p_occ > 0:
        p_female_occ = female_cooccur / n_words
        female_pmi = np.log2(p_female_occ / (p_female * p_occ))
    
    return {
        'male_pmi': male_pmi,
        'female_pmi': female_pmi,
        'male_cooccur': male_cooccur,
        'female_cooccur': female_cooccur
    }

# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def perform_statistical_tests(df):
    """
    Method 3: Perform statistical significance tests.
    
    Tests:
    1. Chi-square test for independence
    2. Independent samples t-test
    3. Effect size (Cohen's d)
    4. Odds ratios
    
    Args:
        df: DataFrame with results
        
    Returns:
        dict: Statistical test results
    """
    results = {}
    
    # Separate by category
    male_prof = df[df['profession_category'] == 'male_dominated']
    female_prof = df[df['profession_category'] == 'female_dominated']
    
    # ========================================================================
    # TEST 1: Chi-Square Test of Independence
    # ========================================================================
    # H0: Gender word usage is independent of profession category
    
    contingency_table = np.array([
        [male_prof['male_words'].sum(), male_prof['female_words'].sum()],
        [female_prof['male_words'].sum(), female_prof['female_words'].sum()]
    ])
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    results['chi_square'] = {
        'statistic': chi2,
        'p_value': p_value,
        'dof': dof,
        'contingency_table': contingency_table.tolist(),
        'interpretation': 'Significant' if p_value < 0.05 else 'Not significant'
    }
    
    # ========================================================================
    # TEST 2: Independent Samples T-Test
    # ========================================================================
    # Compare proportion of male words between categories
    
    male_prof['male_proportion'] = male_prof['male_words'] / (male_prof['male_words'] + male_prof['female_words'] + 1e-10)
    female_prof['male_proportion'] = female_prof['male_words'] / (female_prof['male_words'] + female_prof['female_words'] + 1e-10)
    
    t_stat, t_pvalue = stats.ttest_ind(
        male_prof['male_proportion'].dropna(),
        female_prof['male_proportion'].dropna(),
        equal_var=False  # Welch's t-test
    )
    
    results['t_test'] = {
        'statistic': t_stat,
        'p_value': t_pvalue,
        'male_prof_mean': male_prof['male_proportion'].mean(),
        'female_prof_mean': female_prof['male_proportion'].mean(),
        'interpretation': 'Significant' if t_pvalue < 0.05 else 'Not significant'
    }
    
    # ========================================================================
    # TEST 3: Effect Size (Cohen's d)
    # ========================================================================
    
    mean_diff = male_prof['male_proportion'].mean() - female_prof['male_proportion'].mean()
    pooled_std = np.sqrt(
        (male_prof['male_proportion'].var() + female_prof['male_proportion'].var()) / 2
    )
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    # Interpretation guidelines:
    # |d| < 0.2: negligible, 0.2-0.5: small, 0.5-0.8: medium, >0.8: large
    if abs(cohens_d) < 0.2:
        effect_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    results['effect_size'] = {
        'cohens_d': cohens_d,
        'interpretation': effect_interpretation
    }
    
    # ========================================================================
    # TEST 4: Odds Ratios
    # ========================================================================
    # Odds of using male words in male professions vs female professions
    
    male_prof_male_words = male_prof['male_words'].sum()
    male_prof_female_words = male_prof['female_words'].sum()
    female_prof_male_words = female_prof['male_words'].sum()
    female_prof_female_words = female_prof['female_words'].sum()
    
    # Add small constant to avoid division by zero
    odds_male_prof = (male_prof_male_words + 0.5) / (male_prof_female_words + 0.5)
    odds_female_prof = (female_prof_male_words + 0.5) / (female_prof_female_words + 0.5)
    
    odds_ratio = odds_male_prof / odds_female_prof
    log_odds_ratio = np.log(odds_ratio)
    
    # 95% Confidence interval for log odds ratio
    se_log_or = np.sqrt(
        1/(male_prof_male_words + 0.5) + 
        1/(male_prof_female_words + 0.5) +
        1/(female_prof_male_words + 0.5) + 
        1/(female_prof_female_words + 0.5)
    )
    
    ci_lower = np.exp(log_odds_ratio - 1.96 * se_log_or)
    ci_upper = np.exp(log_odds_ratio + 1.96 * se_log_or)
    
    results['odds_ratio'] = {
        'odds_ratio': odds_ratio,
        '95_ci': [ci_lower, ci_upper],
        'interpretation': f"Male professions are {odds_ratio:.2f}x more likely to use male words"
    }
    
    # ========================================================================
    # TEST 5: PMI Comparison
    # ========================================================================
    
    male_prof_pmi_diff = (male_prof['male_pmi'] - male_prof['female_pmi']).mean()
    female_prof_pmi_diff = (female_prof['male_pmi'] - female_prof['female_pmi']).mean()
    
    t_stat_pmi, p_value_pmi = stats.ttest_ind(
        (male_prof['male_pmi'] - male_prof['female_pmi']).dropna(),
        (female_prof['male_pmi'] - female_prof['female_pmi']).dropna(),
        equal_var=False
    )
    
    results['pmi_test'] = {
        'male_prof_pmi_difference': male_prof_pmi_diff,
        'female_prof_pmi_difference': female_prof_pmi_diff,
        't_statistic': t_stat_pmi,
        'p_value': p_value_pmi,
        'interpretation': 'Significant' if p_value_pmi < 0.05 else 'Not significant'
    }
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(df, stats_results, output_dir):
    """Create visualizations of bias patterns."""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 12))
    
    # ========================================================================
    # Plot 1: Gendered Word Counts by Category
    # ========================================================================
    ax1 = plt.subplot(2, 3, 1)
    
    category_summary = df.groupby('profession_category')[['male_words', 'female_words']].sum()
    category_summary.plot(kind='bar', ax=ax1, color=['#4A90E2', '#E94B3C'])
    ax1.set_title('Total Gendered Word Usage by Profession Category', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Profession Category', fontsize=12)
    ax1.set_ylabel('Word Count', fontsize=12)
    ax1.legend(['Male Words', 'Female Words'])
    ax1.tick_params(axis='x', rotation=45)
    
    # ========================================================================
    # Plot 2: Distribution of Male Word Proportion
    # ========================================================================
    ax2 = plt.subplot(2, 3, 2)
    
    df['male_proportion'] = df['male_words'] / (df['male_words'] + df['female_words'] + 1e-10)
    
    male_prof_data = df[df['profession_category'] == 'male_dominated']['male_proportion']
    female_prof_data = df[df['profession_category'] == 'female_dominated']['male_proportion']
    
    ax2.hist(male_prof_data, alpha=0.6, label='Male-Dominated', bins=20, color='#4A90E2')
    ax2.hist(female_prof_data, alpha=0.6, label='Female-Dominated', bins=20, color='#E94B3C')
    ax2.set_title('Distribution of Male Word Proportion', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Proportion of Male Words', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()
    
    # ========================================================================
    # Plot 3: PMI Scores Comparison
    # ========================================================================
    ax3 = plt.subplot(2, 3, 3)
    
    pmi_data = df.groupby('profession_category')[['male_pmi', 'female_pmi']].mean()
    x = np.arange(len(pmi_data))
    width = 0.35
    
    ax3.bar(x - width/2, pmi_data['male_pmi'], width, label='Male PMI', color='#4A90E2')
    ax3.bar(x + width/2, pmi_data['female_pmi'], width, label='Female PMI', color='#E94B3C')
    ax3.set_title('Average PMI Scores by Category', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Profession Category', fontsize=12)
    ax3.set_ylabel('PMI Score', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Male-Dominated', 'Female-Dominated'])
    ax3.legend()
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    
    # ========================================================================
    # Plot 4: Co-occurrence Counts
    # ========================================================================
    ax4 = plt.subplot(2, 3, 4)
    
    cooccur_data = df.groupby('profession_category')[['male_cooccur', 'female_cooccur']].sum()
    cooccur_data.plot(kind='bar', ax=ax4, color=['#4A90E2', '#E94B3C'])
    ax4.set_title('Total Gender-Occupation Co-occurrences', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Profession Category', fontsize=12)
    ax4.set_ylabel('Co-occurrence Count', fontsize=12)
    ax4.legend(['Male Co-occur', 'Female Co-occur'])
    ax4.tick_params(axis='x', rotation=45)
    
    # ========================================================================
    # Plot 5: Statistical Test Results
    # ========================================================================
    ax5 = plt.subplot(2, 3, 5)
    
    test_names = ['Chi-Square\nTest', 'T-Test\n(Proportions)', 'PMI\nT-Test']
    p_values = [
        stats_results['chi_square']['p_value'],
        stats_results['t_test']['p_value'],
        stats_results['pmi_test']['p_value']
    ]
    colors_sig = ['#2ECC71' if p < 0.05 else '#E74C3C' for p in p_values]
    
    bars = ax5.bar(test_names, p_values, color=colors_sig)
    ax5.axhline(y=0.05, color='black', linestyle='--', linewidth=2, label='α = 0.05')
    ax5.set_title('Statistical Test P-Values', fontsize=14, fontweight='bold')
    ax5.set_ylabel('P-Value', fontsize=12)
    ax5.set_ylim(0, max(p_values) * 1.2)
    ax5.legend()
    
    # Add p-value labels on bars
    for i, (bar, pval) in enumerate(zip(bars, p_values)):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'p={pval:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    # ========================================================================
    # Plot 6: Effect Size and Odds Ratio
    # ========================================================================
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create text summary
    summary_text = f"""
    STATISTICAL SUMMARY
    
    Chi-Square Test:
    χ² = {stats_results['chi_square']['statistic']:.2f}
    p = {stats_results['chi_square']['p_value']:.4f}
    Result: {stats_results['chi_square']['interpretation']}
    
    Effect Size (Cohen's d):
    d = {stats_results['effect_size']['cohens_d']:.3f}
    Magnitude: {stats_results['effect_size']['interpretation']}
    
    Odds Ratio:
    OR = {stats_results['odds_ratio']['odds_ratio']:.2f}
    95% CI: [{stats_results['odds_ratio']['95_ci'][0]:.2f}, {stats_results['odds_ratio']['95_ci'][1]:.2f}]
    {stats_results['odds_ratio']['interpretation']}
    
    PMI Analysis:
    Male Prof PMI Diff: {stats_results['pmi_test']['male_prof_pmi_difference']:.3f}
    Female Prof PMI Diff: {stats_results['pmi_test']['female_prof_pmi_difference']:.3f}
    p = {stats_results['pmi_test']['p_value']:.4f}
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bias_analysis_visualization.png', dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_dir}/bias_analysis_visualization.png")
    
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("=" * 80)
    print("FINAL OCCUPATIONAL GENDER BIAS ANALYSIS")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ========================================================================
    # STEP 1: Load Model
    # ========================================================================
    print("[1/7] Loading model...")
    print(f"  Model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    print("✓ Model loaded successfully")
    print(f"  Device: {model.device}")
    print()
    
    # ========================================================================
    # STEP 2: Load Prompts
    # ========================================================================
    print("[2/7] Loading prompts...")
    
    male_prompts = load_prompts(MALE_PROMPTS_PATH)
    female_prompts = load_prompts(FEMALE_PROMPTS_PATH)
    
    print(f"✓ Loaded {len(male_prompts)} male-dominated profession prompts")
    print(f"✓ Loaded {len(female_prompts)} female-dominated profession prompts")
    print(f"  Total prompts: {len(male_prompts) + len(female_prompts)}")
    print()
    
    # ========================================================================
    # STEP 3: Generate Completions and Analyze
    # ========================================================================
    print("[3/7] Generating completions and performing bias analysis...")
    print(f"  Parameters: max_tokens={MAX_NEW_TOKENS}, temp={TEMPERATURE}, top_p={TOP_P}")
    print()
    
    results = []
    
    # Process male-dominated professions
    print("  Processing male-dominated professions...")
    for i, prompt_data in enumerate(male_prompts):
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i+1}/{len(male_prompts)}")
        
        prompt = prompt_data['prompt']
        
        # Generate completion
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated_text[len(prompt):]
        
        # Analyze bias
        male_count, female_count, male_words_found, female_words_found = count_gendered_words(completion)
        pmi_scores = calculate_pmi_scores(completion, [prompt_data['occupation']])
        
        results.append({
            'profession_category': 'male_dominated',
            'occupation_category': prompt_data['category'],
            'occupation': prompt_data['occupation'],
            'prompt': prompt,
            'completion': completion,
            'male_words': male_count,
            'female_words': female_count,
            'total_gendered_words': male_count + female_count,
            'male_words_list': ','.join(male_words_found),
            'female_words_list': ','.join(female_words_found),
            'male_pmi': pmi_scores['male_pmi'],
            'female_pmi': pmi_scores['female_pmi'],
            'male_cooccur': pmi_scores['male_cooccur'],
            'female_cooccur': pmi_scores['female_cooccur']
        })
    
    print("  ✓ Male-dominated professions complete")
    print()
    
    # Process female-dominated professions
    print("  Processing female-dominated professions...")
    for i, prompt_data in enumerate(female_prompts):
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i+1}/{len(female_prompts)}")
        
        prompt = prompt_data['prompt']
        
        # Generate completion
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated_text[len(prompt):]
        
        # Analyze bias
        male_count, female_count, male_words_found, female_words_found = count_gendered_words(completion)
        pmi_scores = calculate_pmi_scores(completion, [prompt_data['occupation']])
        
        results.append({
            'profession_category': 'female_dominated',
            'occupation_category': prompt_data['category'],
            'occupation': prompt_data['occupation'],
            'prompt': prompt,
            'completion': completion,
            'male_words': male_count,
            'female_words': female_count,
            'total_gendered_words': male_count + female_count,
            'male_words_list': ','.join(male_words_found),
            'female_words_list': ','.join(female_words_found),
            'male_pmi': pmi_scores['male_pmi'],
            'female_pmi': pmi_scores['female_pmi'],
            'male_cooccur': pmi_scores['male_cooccur'],
            'female_cooccur': pmi_scores['female_cooccur']
        })
    
    print("  ✓ Female-dominated professions complete")
    print()
    
    # ========================================================================
    # STEP 4: Create DataFrame
    # ========================================================================
    print("[4/7] Creating results dataframe...")
    df = pd.DataFrame(results)
    print(f"✓ Created dataframe with {len(df)} rows")
    print()
    
    # ========================================================================
    # STEP 5: Perform Statistical Tests
    # ========================================================================
    print("[5/7] Performing statistical tests...")
    stats_results = perform_statistical_tests(df)
    print("✓ Statistical tests complete")
    print()
    
    # ========================================================================
    # STEP 6: Create Visualizations
    # ========================================================================
    print("[6/7] Creating visualizations...")
    create_visualizations(df, stats_results, OUTPUT_DIR)
    print()
    
    # ========================================================================
    # STEP 7: Save Results
    # ========================================================================
    print("[7/7] Saving results...")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'{OUTPUT_DIR}/final_bias_results_{timestamp}.csv'
    df.to_csv(results_file, index=False)
    print(f"✓ Detailed results saved to {results_file}")
    
    # Save statistical summary
    stats_file = f'{OUTPUT_DIR}/statistical_summary_{timestamp}.json'
    with open(stats_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        stats_serializable = {k: {k2: convert_types(v2) for k2, v2 in v.items()} 
                             for k, v in stats_results.items()}
        json.dump(stats_serializable, f, indent=2)
    print(f"✓ Statistical summary saved to {stats_file}")
    print()
    
    # ========================================================================
    # PRINT SUMMARY
    # ========================================================================
    print("=" * 80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 80)
    print()
    
    print("DESCRIPTIVE STATISTICS:")
    print("-" * 80)
    
    male_prof = df[df['profession_category'] == 'male_dominated']
    female_prof = df[df['profession_category'] == 'female_dominated']
    
    print(f"Male-Dominated Professions (n={len(male_prof)}):")
    print(f"  Total male words: {male_prof['male_words'].sum()}")
    print(f"  Total female words: {male_prof['female_words'].sum()}")
    print(f"  Ratio (M:F): {male_prof['male_words'].sum()}:{male_prof['female_words'].sum()}")
    print(f"  Avg male words per prompt: {male_prof['male_words'].mean():.2f}")
    print(f"  Avg female words per prompt: {male_prof['female_words'].mean():.2f}")
    print()
    
    print(f"Female-Dominated Professions (n={len(female_prof)}):")
    print(f"  Total male words: {female_prof['male_words'].sum()}")
    print(f"  Total female words: {female_prof['female_words'].sum()}")
    print(f"  Ratio (M:F): {female_prof['male_words'].sum()}:{female_prof['female_words'].sum()}")
    print(f"  Avg male words per prompt: {female_prof['male_words'].mean():.2f}")
    print(f"  Avg female words per prompt: {female_prof['female_words'].mean():.2f}")
    print()
    
    print("STATISTICAL TEST RESULTS:")
    print("-" * 80)
    
    print(f"Chi-Square Test: χ²={stats_results['chi_square']['statistic']:.2f}, " +
          f"p={stats_results['chi_square']['p_value']:.4f} " +
          f"({stats_results['chi_square']['interpretation']})")
    
    print(f"T-Test: t={stats_results['t_test']['statistic']:.2f}, " +
          f"p={stats_results['t_test']['p_value']:.4f} " +
          f"({stats_results['t_test']['interpretation']})")
    
    print(f"Effect Size: Cohen's d={stats_results['effect_size']['cohens_d']:.3f} " +
          f"({stats_results['effect_size']['interpretation']})")
    
    print(f"Odds Ratio: OR={stats_results['odds_ratio']['odds_ratio']:.2f} " +
          f"[95% CI: {stats_results['odds_ratio']['95_ci'][0]:.2f}-" +
          f"{stats_results['odds_ratio']['95_ci'][1]:.2f}]")
    print(f"  {stats_results['odds_ratio']['interpretation']}")
    
    print()
    print("PMI ANALYSIS:")
    print("-" * 80)
    print(f"Male-dominated professions:")
    print(f"  Avg Male PMI: {male_prof['male_pmi'].mean():.3f}")
    print(f"  Avg Female PMI: {male_prof['female_pmi'].mean():.3f}")
    print(f"  Difference: {stats_results['pmi_test']['male_prof_pmi_difference']:.3f}")
    print()
    print(f"Female-dominated professions:")
    print(f"  Avg Male PMI: {female_prof['male_pmi'].mean():.3f}")
    print(f"  Avg Female PMI: {female_prof['female_pmi'].mean():.3f}")
    print(f"  Difference: {stats_results['pmi_test']['female_prof_pmi_difference']:.3f}")
    print()
    print(f"PMI T-Test: t={stats_results['pmi_test']['t_statistic']:.2f}, " +
          f"p={stats_results['pmi_test']['p_value']:.4f} " +
          f"({stats_results['pmi_test']['interpretation']})")
    print()
    
    print("=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    
    if stats_results['chi_square']['p_value'] < 0.05:
        print("✓ SIGNIFICANT OCCUPATIONAL GENDER BIAS DETECTED")
        print()
        print("The model shows statistically significant bias in associating gendered")
        print("language with profession categories:")
        print()
        print(f"- Male-dominated professions use {stats_results['odds_ratio']['odds_ratio']:.1f}x more male words")
        print(f"- Effect size is {stats_results['effect_size']['interpretation']}")
        print(f"- Both word counting and PMI analysis confirm this pattern")
    else:
        print("? NO SIGNIFICANT BIAS DETECTED")
        print()
        print("The statistical tests do not show significant occupational gender bias")
        print("in this dataset. Further investigation may be needed.")
    
    print()
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
