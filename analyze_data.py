import re
from collections import defaultdict

# Read background knowledge
bk = {}
with open('ilp/bk.pl') as f:
    for line in f:
        m = re.match(r'enemy_dist\((\S+),(\S+)\)\.', line)
        if m:
            state, val = m.groups()
            bk.setdefault(state, {})['enemy'] = val
        m = re.match(r'gap_dist\((\S+),(\S+)\)\.', line)
        if m:
            state, val = m.groups()
            bk.setdefault(state, {})['gap'] = val
        m = re.match(r'on_ground\((\S+),(\S+)\)\.', line)
        if m:
            state, val = m.groups()
            bk.setdefault(state, {})['ground'] = val

# Read examples
examples = []
with open('ilp/exs.pl') as f:
    for line in f:
        m = re.match(r'(pos|neg)\(good_action\((\S+),(\S+)\)\)\.', line)
        if m:
            label, state, action = m.groups()
            if state in bk:
                features = (bk[state].get('enemy'), bk[state].get('gap'), bk[state].get('ground'), action)
                examples.append((features, label))

# Group by features and check for contradictions
feature_labels = defaultdict(lambda: {'pos': 0, 'neg': 0})
for features, label in examples:
    feature_labels[features][label] += 1

print('Looking for feature combinations with both pos and neg labels:')
contradictions = [(f, c) for f, c in feature_labels.items() if c['pos'] > 0 and c['neg'] > 0]
print(f'Found {len(contradictions)} contradictory patterns')
for (enemy, gap, ground, action), counts in contradictions[:10]:
    print(f'  enemy={enemy}, gap={gap}, ground={ground}, action={action} -> pos={counts["pos"]}, neg={counts["neg"]}')

print()
print('Positive-only patterns:')
pos_only = [(f, c) for f, c in feature_labels.items() if c['pos'] > 0 and c['neg'] == 0]
for (enemy, gap, ground, action), counts in pos_only[:10]:
    print(f'  enemy={enemy}, gap={gap}, ground={ground}, action={action} -> pos={counts["pos"]}')

print()
print('Summary:')
print(f'  Total unique feature combinations: {len(feature_labels)}')
print(f'  Contradictory: {len(contradictions)}')
print(f'  Positive-only: {len(pos_only)}')
print(f'  Negative-only: {len(feature_labels) - len(contradictions) - len(pos_only)}')

print()
print('Negative-only patterns:')
neg_only = [(f, c) for f, c in feature_labels.items() if c['pos'] == 0 and c['neg'] > 0]
for (enemy, gap, ground, action), counts in neg_only[:15]:
    print(f'  enemy={enemy}, gap={gap}, ground={ground}, action={action} -> neg={counts["neg"]}')
