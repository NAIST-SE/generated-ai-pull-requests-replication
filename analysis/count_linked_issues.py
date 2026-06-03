import pandas as pd
import re
from pathlib import Path
from typing import Set

def extract_issue_references(body: str) -> Set[str]:
    """Extract JabRef issue references from PR body (ignore external references)"""
    if pd.isna(body) or not str(body).strip():
        return set()
    
    body = str(body)
    issue_numbers = set()
    
    # JabRef specific issue patterns (more restrictive to avoid external issues)
    issue_patterns = [
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
    external_patterns = [
        r'https://github-redirect\.dependabot\.com/',
        r'https://github\.com/(?!JabRef/jabref|koppor/jabref)',  # Non-JabRef repos
    ]
    
    # Extract issue numbers using JabRef-specific patterns only
    for pattern in issue_patterns:
        matches = re.finditer(pattern, body, re.IGNORECASE)
        for match in matches:
            issue_num = match.group(1)
            issue_numbers.add(issue_num)
            
    return issue_numbers

def is_template_reference(body: str) -> bool:
    """Check if the body only contains template issue references"""
    if pd.isna(body) or not str(body).strip():
        return True
    
    body = str(body)
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

def count_linked_issues():
    base_path = Path(__file__).parent.parent / "datasets"
    input_path = base_path / "jabref-prs.csv"
    
    if not input_path.exists():
        print(f"Error: File not found at {input_path}")
        return

    print(f"Reading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Total PRs in dataset: {len(df)}")
    
    # Apply extraction
    print("Extracting issue references...")
    df['issue_references'] = df['body'].apply(lambda x: extract_issue_references(x))
    
    # Check for templates
    print("Checking for template/placeholder content...")
    df['is_template'] = df['body'].apply(lambda x: is_template_reference(x))
    
    # Valid PRs have issues AND are not just templates
    df['has_issue'] = df['issue_references'].apply(lambda x: len(x) > 0)
    
    # Count raw matches
    raw_count = df['has_issue'].sum()
    
    # Count after filtering templates
    # We only exclude if it has an issue BUT is marked as a template
    df['valid_linked_issue'] = df['has_issue'] & (~df['is_template'])
    
    valid_count = df['valid_linked_issue'].sum()
    
    print("-" * 50)
    print(f"PRs with ANY linked issue pattern: {raw_count} ({(raw_count/len(df))*100:.2f}%)")
    print(f"PRs with VALID linked issues (excluding templates): {valid_count} ({(valid_count/len(df))*100:.2f}%)")
    print(f"Excluded as templates/placeholders: {raw_count - valid_count}")
    print("-" * 50)

if __name__ == "__main__":
    count_linked_issues()
