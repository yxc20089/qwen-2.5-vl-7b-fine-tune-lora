# Full Test Dataset Evaluation
import torch
import json
import re
from tqdm.auto import tqdm
from PIL import Image

def extract_answer(text, question_type='multi-choice'):
    """
    Extract answer from model output.
    For multi-choice: look for single letter A, B, C, D
    For free-form: return the full text
    """
    if question_type == 'multi-choice':
        # Look for patterns like "A", "B", "C", "D" (case insensitive)
        # Try to find the last occurrence to get the final answer
        matches = re.findall(r'\b([A-D])\b', text.upper())
        if matches:
            return matches[-1]  # Return last match
        return None
    else:
        # For free-form, return cleaned text
        return text.strip()

def evaluate_model(model, processor, test_data, max_samples=None):
    """
    Evaluate model on test dataset.

    Args:
        model: Fine-tuned model
        processor: Model processor
        test_data: List of test samples
        max_samples: Maximum samples to evaluate (None for all)

    Returns:
        Dictionary with results and metrics
    """
    model.eval()
    results = []
    correct = 0
    total = 0

    # Limit samples if specified
    samples_to_eval = test_data[:max_samples] if max_samples else test_data

    print(f"Evaluating on {len(samples_to_eval)} samples...")

    for idx, sample in enumerate(tqdm(samples_to_eval, desc="Evaluating")):
        # Extract data from conversation format
        test_image = sample['messages'][0]['content'][0]['image']
        test_question = sample['messages'][0]['content'][1]['text']
        expected_answer = sample['messages'][1]['content'][0]['text']

        # Determine question type (try to infer from question)
        question_type = 'multi-choice' if 'Choices:' in test_question or any(opt in test_question for opt in ['A:', 'B:', 'C:', 'D:']) else 'free-form'

        # Prepare input
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},
                    {"type": "text", "text": test_question}
                ]
            }
        ]

        text_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = processor(
            text=[text_prompt],
            images=[test_image],
            return_tensors="pt",
            padding=True
        ).to(model.device)

        # Generate response
        try:
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.0,  # Use greedy decoding for evaluation
                )

            # Decode response
            generated_text = processor.batch_decode(
                output,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]

            # Extract answer from generation
            predicted_answer = extract_answer(generated_text, question_type)

            # Check correctness
            is_correct = False
            if question_type == 'multi-choice':
                # For multi-choice, compare extracted letters
                expected_letter = extract_answer(expected_answer, 'multi-choice')
                if expected_letter is None:
                    expected_letter = expected_answer.strip().upper()
                is_correct = (predicted_answer == expected_letter)
            else:
                # For free-form, check if expected answer is in prediction
                is_correct = (expected_answer.lower() in generated_text.lower())

            if is_correct:
                correct += 1
            total += 1

            # Store result
            results.append({
                'index': idx,
                'question': test_question[:200] + '...' if len(test_question) > 200 else test_question,
                'expected_answer': expected_answer,
                'predicted_answer': predicted_answer if predicted_answer else generated_text[:100],
                'full_response': generated_text,
                'question_type': question_type,
                'correct': is_correct
            })

        except Exception as e:
            print(f"\nError on sample {idx}: {e}")
            results.append({
                'index': idx,
                'question': test_question[:200],
                'expected_answer': expected_answer,
                'predicted_answer': 'ERROR',
                'full_response': str(e),
                'question_type': question_type,
                'correct': False
            })

        # Print progress every 10 samples
        if (idx + 1) % 10 == 0:
            current_acc = (correct / total * 100) if total > 0 else 0
            print(f"\nProgress: {idx + 1}/{len(samples_to_eval)} | Accuracy: {current_acc:.2f}%")

    # Calculate final metrics
    accuracy = (correct / total * 100) if total > 0 else 0

    # Calculate per-type accuracy
    mc_correct = sum(1 for r in results if r['question_type'] == 'multi-choice' and r['correct'])
    mc_total = sum(1 for r in results if r['question_type'] == 'multi-choice')
    ff_correct = sum(1 for r in results if r['question_type'] == 'free-form' and r['correct'])
    ff_total = sum(1 for r in results if r['question_type'] == 'free-form')

    metrics = {
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'multi_choice_accuracy': (mc_correct / mc_total * 100) if mc_total > 0 else 0,
        'multi_choice_count': mc_total,
        'free_form_accuracy': (ff_correct / ff_total * 100) if ff_total > 0 else 0,
        'free_form_count': ff_total
    }

    return results, metrics

# Run evaluation on validation set
print("=" * 80)
print("FULL VALIDATION SET EVALUATION")
print("=" * 80)

# Evaluate (set max_samples=10 for quick test, None for full dataset)
results, metrics = evaluate_model(
    model,
    processor,
    val_data,
    max_samples=None  # Set to 10 for quick test, None for full evaluation
)

# Print metrics
print("\n" + "=" * 80)
print("EVALUATION RESULTS")
print("=" * 80)
print(f"Total Samples: {metrics['total_samples']}")
print(f"Correct: {metrics['correct']}")
print(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
print(f"\nMulti-Choice Accuracy: {metrics['multi_choice_accuracy']:.2f}% ({metrics['multi_choice_count']} samples)")
print(f"Free-Form Accuracy: {metrics['free_form_accuracy']:.2f}% ({metrics['free_form_count']} samples)")
print("=" * 80)

# Save results to JSON file
output_file = "./evaluation_results.json"
with open(output_file, 'w') as f:
    json.dump({
        'metrics': metrics,
        'results': results
    }, f, indent=2)

print(f"\n✅ Results saved to: {output_file}")

# Show some examples
print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS (First 5)")
print("=" * 80)
for i, result in enumerate(results[:5]):
    print(f"\nSample {i+1}:")
    print(f"Question: {result['question']}")
    print(f"Expected: {result['expected_answer']}")
    print(f"Predicted: {result['predicted_answer']}")
    print(f"Correct: {'✓' if result['correct'] else '✗'}")
    print("-" * 40)
