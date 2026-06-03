#!/usr/bin/env python3
"""
Filter and sample PRs from the raw JabRef dump.

Objectives:
- Filter out PRs without any referenced issues AND too short descriptions
- Stratified sampling by change size (small, medium, large)
- Select 100 representative PRs
- Extract and validate issue references from PR bodies
- Exclude template issues and external repository references
- Filter out descriptions that are too short for quality generation

Input: datasets/jabref-prs.csv  (download from OSF: https://osf.io/aq5rj/)
Output: datasets/selected_prs.csv
"""

import pandas as pd
import re
import numpy as np
import datetime
from pathlib import Path
from typing import List, Set, Tuple
import logging

# Configure logging
LOG_PATH = Path(__file__).parent.parent / "datasets" / "filter_and_sample_output.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_activity(activity: str):
    """Log activity with timestamp"""
    logger.info(activity)

class PRPreprocessor:
    def __init__(self, input_path: str, output_path: str, min_description_length: int = 75, min_word_count: int = 10):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.min_description_length = min_description_length
        self.min_word_count = min_word_count
        
        # Common template issue patterns to exclude
        self.template_issues = {
            '#[0-9]+',  # Generic placeholder like #123 in templates
            'issue #[0-9]+',  # Generic "issue #123"
            'issues/[0-9]+',  # Generic "issues/123"
        }
        
        # JabRef specific issue patterns (more restrictive to avoid external issues)
        self.issue_patterns = [
            # Direct JabRef issue references with action words
            r'(?:fixes?|resolves?|closes?)\s+#(\d+)',
            r'(?:fixes?|resolves?|closes?)\s+issue\s+#(\d+)',
            r'(?:fixes?|resolves?|closes?)\s+jabref[/#](\d+)',
            
            # JabRef GitHub URLs only
            r'https://github\.com/(?:JabRef|koppor)/jabref/issues/(\d+)',
            
            # More restrictive #number references (with better context)
            r'(?:^|[.\s,;])\s*#(\d+)(?=\s|$|[,.])',  # Only in proper contexts
            
            # JabRef-specific issue mentions
            r'(?:JabRef\s+issue\s+|jabref[/#])(\d+)',
        ]
        
        # Dependency/external issue patterns to exclude
        self.external_patterns = [
            r'https://github-redirect\.dependabot\.com/',
            r'https://github\.com/(?!JabRef/jabref|koppor/jabref)',  # Non-JabRef repos
        ]

    def extract_issue_references(self, body: str) -> Set[str]:
        """Extract JabRef issue references from PR body (ignore external references)"""
        if pd.isna(body) or not body.strip():
            return set()
        
        issue_numbers = set()
        
        # Extract issue numbers using JabRef-specific patterns only
        # Skip external references but don't exclude the entire PR
        for pattern in self.issue_patterns:
            matches = re.finditer(pattern, body, re.IGNORECASE)
            for match in matches:
                issue_num = match.group(1)
                issue_numbers.add(issue_num)
        
        # Log if we found external patterns but continue processing
        for pattern in self.external_patterns:
            if re.search(pattern, body, re.IGNORECASE):
                log_activity(f"Found external issue pattern in PR body: {pattern} - EXTRACTING JABREF ISSUES ONLY")
                break
        
        return issue_numbers

    def is_template_reference(self, body: str) -> bool:
        """Check if the body only contains template issue references"""
        if pd.isna(body) or not body.strip():
            return True
        
        # Look for common template patterns
        template_indicators = [
            r'^\s*<!--.*-->',  # Comment blocks
            r'checklist',
            r'template',
            r'placeholder',
            r'example',
            r'\[x\]|\[ \]',  # Checkbox patterns
        ]
        
        for pattern in template_indicators:
            if re.search(pattern, body, re.IGNORECASE | re.MULTILINE):
                return True
        
        return False

    def filter_description_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter PRs with adequate description length"""
        log_activity(f"Filtering PRs with descriptions >= {self.min_description_length} chars and >= {self.min_word_count} words...")
        
        # Calculate description metrics
        df['body_length'] = df['body'].fillna('').astype(str).str.len()
        df['word_count'] = df['body'].fillna('').astype(str).str.split().str.len()
        
        # Filter by both character count and word count
        filtered_df = df[
            (df['body_length'] >= self.min_description_length) & 
            (df['word_count'] >= self.min_word_count)
        ].copy()
        
        log_activity(f"Filtered from {len(df)} to {len(filtered_df)} PRs based on description length")
        return filtered_df

    def filter_prs_with_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter PRs that have at least one valid JabRef issue reference"""
        log_activity("Starting to filter PRs with JabRef issue references...")
        
        # Add issue references column (extract JabRef issues, ignore external)
        df['issue_references'] = df['body'].apply(
            lambda x: ','.join(sorted(self.extract_issue_references(x))) if self.extract_issue_references(x) else ''
        )
        
        # Filter: must have at least one JabRef issue reference
        prs_with_issues = df[df['issue_references'].str.len() > 0].copy()
        
        # Filter out template-only PRs
        prs_with_issues = prs_with_issues[
            ~prs_with_issues['body'].apply(self.is_template_reference)
        ]
        
        log_activity(f"Found {len(prs_with_issues)} PRs with valid JabRef issue references out of {len(df)} total PRs")
        
        return prs_with_issues

    def stratified_sampling(self, df: pd.DataFrame, samples_per_tier: int = 50) -> pd.DataFrame:
        """Perform stratified sampling based on sorted total churn tiers"""
        log_activity("Performing stratified sampling based on total churn tiers...")
        
        # 1. Filter: Confirm all PRs have at least one modified file and non-zero code churn
        df['total_churn'] = df['additions_count'].fillna(0) + df['deletions_count'].fillna(0)
        
        df = df[
            (df['changed_files_count'] > 0) & 
            (df['total_churn'] > 0)
        ].copy()
        log_activity(f"Filtered to {len(df)} PRs with changed files > 0 and total churn > 0")

        # 2. Sort: Sort the PRs in ascending order based on total_churn
        df_sorted = df.sort_values('total_churn', ascending=True)
        
        # 3. Stratify: Split the sorted dataset into 3 equal tiers
        n = len(df_sorted)
        if n < 3:
             # Fallback if very few PRs
             log_activity("Not enough PRs to split into tiers. Returning all.")
             return df_sorted

        # Indices for splitting
        # Bottom tier: small 33%
        # Middle tier: middle 34%
        # Top tier: top 33%
        idx1 = int(n * 0.33)
        idx2 = int(n * (0.33 + 0.34))
        
        bottom_tier = df_sorted.iloc[:idx1].copy()
        middle_tier = df_sorted.iloc[idx1:idx2].copy()
        top_tier = df_sorted.iloc[idx2:].copy()
        
        bottom_tier['change_size'] = 'small'
        middle_tier['change_size'] = 'medium'
        top_tier['change_size'] = 'large'
        
        log_activity(f"Tier sizes - Small: {len(bottom_tier)}, Medium: {len(middle_tier)}, Large: {len(top_tier)}")
        
        # 4. Sample: From each tier, randomly sample 50 PRs using a fixed random seed
        sampled_dfs = []
        for tier_name, tier_df in [('small', bottom_tier), ('medium', middle_tier), ('large', top_tier)]:
            if len(tier_df) == 0:
                log_activity(f"Warning: {tier_name} tier is empty")
                continue
                
            sample_size = min(samples_per_tier, len(tier_df))
            sampled = tier_df.sample(n=sample_size, random_state=42)
            sampled_dfs.append(sampled)
            log_activity(f"Sampled {sample_size} PRs from {tier_name} tier")
            
        result = pd.concat(sampled_dfs, ignore_index=True)
        log_activity(f"Final sample size: {len(result)} PRs")
        
        return result

    def process(self) -> pd.DataFrame:
        """Main processing pipeline"""
        log_activity(f"Starting preprocessing pipeline...")
        log_activity(f"Input file: {self.input_path}")
        log_activity(f"Output file: {self.output_path}")
        log_activity(f"Minimum description length: {self.min_description_length} characters")
        log_activity(f"Minimum word count: {self.min_word_count} words")
        
        # Load data
        log_activity("Loading PR dataset...")
        df = pd.read_csv(self.input_path)
        log_activity(f"Loaded {len(df)} PRs")
        
        # Step 1: Filter PRs with adequate description length
        length_filtered_df = self.filter_description_length(df)
        
        if len(length_filtered_df) == 0:
            raise ValueError("No PRs found with adequate description length!")
        
        # Step 2: Filter PRs with JabRef issue references
        issue_filtered_df = self.filter_prs_with_issues(length_filtered_df)
        
        if len(issue_filtered_df) == 0:
            raise ValueError("No PRs found with both adequate description length and JabRef issue references!")
        
        # Step 3: Perform stratified sampling
        sampled_df = self.stratified_sampling(issue_filtered_df)
        
        # Add metadata columns for tracking
        sampled_df['selected_at'] = datetime.datetime.now().isoformat()
        sampled_df['selection_method'] = 'stratified_sampling_by_change_size'
        
        # Save results
        log_activity(f"Saving {len(sampled_df)} selected PRs to {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        sampled_df.to_csv(self.output_path, index=False)
        
        # Log final statistics
        self.log_final_stats(sampled_df)
        
        return sampled_df

    def log_final_stats(self, df: pd.DataFrame):
        """Log final statistics about the selected PRs"""
        log_activity("=== FINAL SELECTION STATISTICS ===")
        log_activity(f"Total selected PRs: {len(df)}")
        
        # Change size distribution
        size_dist = df['change_size'].value_counts()
        log_activity(f"Change size distribution: {size_dist.to_dict()}")
        
        # Issue reference statistics
        issue_counts = df['issue_references'].apply(lambda x: len(x.split(',')) if x else 0)
        log_activity(f"Issue references per PR - Mean: {issue_counts.mean():.2f}, Median: {issue_counts.median()}")
        
        # Change metrics
        log_activity(f"Files changed - Mean: {df['changed_files_count'].mean():.2f}, Range: {df['changed_files_count'].min()}-{df['changed_files_count'].max()}")
        log_activity(f"Total changes - Mean: {(df['additions_count'] + df['deletions_count']).mean():.2f}")
        
        log_activity("=== END STATISTICS ===")

def main():
    """Main entry point"""
    # Define paths
    base_path = Path(__file__).parent.parent
    input_path = base_path / "datasets" / "jabref-prs.csv"
    output_path = base_path / "datasets" / "selected_prs.csv"
    
    # Initialize and run preprocessor
    preprocessor = PRPreprocessor(str(input_path), str(output_path))
    
    try:
        result = preprocessor.process()
        log_activity(f"✅ Successfully completed preprocessing! Selected {len(result)} PRs.")
        print(f"✅ Successfully selected {len(result)} PRs and saved to {output_path}")
        
        # Display sample results
        print("\n📊 Sample of selected PRs:")
        print(result[['pull_number', 'title', 'change_size', 'issue_references']].head())
        
    except Exception as e:
        log_activity(f"❌ Error during preprocessing: {str(e)}")
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()