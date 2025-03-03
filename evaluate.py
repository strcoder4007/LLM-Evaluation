from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from dotenv import load_dotenv
import requests
import json
from typing import List, Dict

load_dotenv()

RAG_ENDPOINT = "http://localhost:5009/ambedkar"

def create_test_cases() -> List[Dict]:
    with open("test_cases.txt", "r") as file:
        test_cases = json.load(file)
    return test_cases

correctness_metric = GEval(
    name="Correctness",
    criteria="""
    You are evaluating responses about the Indian Constitution and Dr. Ambedkar's work. Compare the actual output with the expected output and score based on:

    1. Factual Accuracy (40% of score):
    - Give 0.4 if all facts match the expected output exactly
    - Give 0.3 if most facts match but with minor differences
    - Give 0.2 if there are some factual inconsistencies
    - Give 0.1 if there are major factual errors
    - Give 0.0 if completely incorrect

    2. Completeness (30% of score):
    - Give 0.3 if all key points are covered
    - Give 0.2 if most key points are covered
    - Give 0.1 if some key points are missing
    - Give 0.0 if major points are missing

    3. Role Consistency (15% of score):
    - Give 0.15 if perfect Dr. Ambedkar perspective maintained
    - Give 0.1 if mostly consistent
    - Give 0.05 if somewhat inconsistent
    - Give 0.0 if completely inconsistent

    4. Format Compliance (15% of score):
    - Give 0.15 if all format requirements met
    - Give 0.1 if minor format issues
    - Give 0.05 if major format issues
    - Give 0.0 if format completely wrong

    Calculate final score by adding all components (max 1.0).

    Example scoring:
    Perfect match = 0.4 + 0.3 + 0.15 + 0.15 = 1.0
    Good match with minor issues = 0.3 + 0.2 + 0.1 + 0.1 = 0.7
    Poor match = 0.1 + 0.1 + 0.05 + 0.05 = 0.3

    Return ONLY a decimal number between 0 and 1, representing the final score.
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5
)

def query_rag_system(input_query: str) -> str:
    try:
        response = requests.post(
            RAG_ENDPOINT,
            json={"query": input_query},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()["answer"]
    except Exception as e:
        print(f"Error querying RAG system: {str(e)}")
        return ""

def evaluate_test_case(test_case: Dict, index: int) -> Dict:
    print(f"\nProcessing Test Case {index}")
    print(f"Input: {test_case['input']}")
    
    actual_output = query_rag_system(test_case['input'])
    print(f"Actual Output: {actual_output}")
    print(f"Expected Output: {test_case['expected_output']}")

    try:
        llm_test_case = LLMTestCase(
            input=test_case['input'],
            actual_output=actual_output,
            expected_output=test_case['expected_output']
        )
        
        correctness_metric.measure(llm_test_case)
        score = correctness_metric.score
        passed = score >= correctness_metric.threshold if score is not None else False

        return {
            'test_case': index,
            'input': test_case['input'],
            'actual_output': actual_output,
            'expected_output': test_case['expected_output'],
            'score': score,
            'passed': passed,
            'error': None
        }
    except Exception as e:
        print(f"Error in test case {index}: {str(e)}")
        return {
            'test_case': index,
            'input': test_case['input'],
            'actual_output': actual_output,
            'expected_output': test_case['expected_output'],
            'score': 0.0,
            'passed': False,
            'error': str(e)
        }

def run_evaluation():
    test_cases = create_test_cases()
    results = []

    print("Starting evaluation...")
    print("-" * 50)

    for i, test_case in enumerate(test_cases, 1):
        result = evaluate_test_case(test_case, i)
        results.append(result)
        
        print(f"\nTest Case {i} Results:")
        print(f"Score: {result['score']:.2f}")
        print(f"Passed: {result['passed']}")
        if result['error']:
            print(f"Error: {result['error']}")
        print("-" * 30)

    passed_tests = sum(1 for r in results if r['passed'])
    total_score = sum(r['score'] for r in results)
    average_score = total_score / len(results) if results else 0

    print("\nEvaluation Summary")
    print("-" * 50)
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(results) - passed_tests}")
    print(f"Success Rate: {(passed_tests/len(results))*100:.2f}%")
    print(f"Average Score: {average_score:.2f}")
    return results

if __name__ == "__main__":
    results = run_evaluation()