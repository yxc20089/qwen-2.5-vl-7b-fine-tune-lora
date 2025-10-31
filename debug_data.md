# Debug Instructions

Add this cell after loading the data to inspect what's being created:

```python
# Debug: Check for image tokens in the text
import re

sample = raw_data[0]
question = sample['messages'][0]['content'][0]['text']

print("=" * 80)
print("DEBUGGING DATA FORMAT")
print("=" * 80)

# Check for various image token patterns
patterns_to_check = [
    '<|image_pad|>',
    '<|vision_start|>',
    '<|vision_end|>',
    '<image>',
    '</image>',
    'image_pad',
    'vision',
]

print(f"\nFull question text:\n{question}\n")
print("-" * 80)

for pattern in patterns_to_check:
    count = question.count(pattern)
    if count > 0:
        print(f"Found '{pattern}': {count} times")

print("\n" + "=" * 80)
print(f"Number of images in sample: {len(sample['images'])}")
print("=" * 80)
```

Run this and share the output so we can see what tokens are in the question text.
